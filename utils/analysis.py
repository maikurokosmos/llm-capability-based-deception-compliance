import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from utils.probe import train_linear_probe


def reduce_activations_pca(
    activations: np.ndarray,
    n_components: int,
) -> dict:
    """
    Apply PCA independently per layer to reduce hidden dimension.

    Parameters
    ----------
    activations  : (n_samples, n_layers, hidden_dim)
    n_components : target number of dimensions after PCA

    Returns
    -------
    dict with keys:
        activations   : np.ndarray (n_samples, n_layers, n_components), float32
        components    : np.ndarray (n_layers, n_components, hidden_dim) — eigenvectors per layer
        explained_var : np.ndarray (n_layers,) — total explained variance ratio per layer
    """
    n_samples, n_layers, hidden_dim = activations.shape
    reduced    = np.zeros((n_samples, n_layers, n_components), dtype=np.float32)
    components = np.zeros((n_layers, n_components, hidden_dim), dtype=np.float32)
    explained_var = np.zeros(n_layers)

    for layer_idx in range(n_layers):
        pca = PCA(n_components=n_components, random_state=42)
        reduced[:, layer_idx, :]    = pca.fit_transform(activations[:, layer_idx, :]).astype(np.float32)
        components[layer_idx]       = pca.components_.astype(np.float32)
        explained_var[layer_idx]    = pca.explained_variance_ratio_.sum()

    return {
        "activations":   reduced,
        "components":    components,
        "explained_var": explained_var,
    }


def select_pca_k(
    activations: np.ndarray,
    labels: np.ndarray,
    k_values: list[int],
    output_path: str | Path = "outputs/pca_reduction_k_selection_results.csv",
) -> pd.DataFrame:
    """
    Select optimal PCA components k by scanning k_values across 3 representative layers
    (25%, 50%, 75% of total depth). For each (layer, k) combination, fits PCA once at
    max(k_values) and slices, then trains a linear probe to record 4 metrics.

    Parameters
    ----------
    activations : (n_samples, n_layers, hidden_dim)
    labels      : (n_samples,) — string labels
    k_values    : list of ints, e.g. [16, 32, 64, 128, 256, 512]
    output_path : path to save results CSV

    Returns
    -------
    pd.DataFrame with columns:
        layer_idx, layer_pct, k, variance_explained,
        f1_macro_val, f1_macro_train, f1_gap, train_time_sec
    """
    output_path = Path(output_path)
    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"[skip] Already complete ({len(df)} rows): {output_path.name}")
        return df

    n_samples, n_layers, hidden_dim = activations.shape
    max_k = max(k_values)

    fracs = [0.25, 0.50, 0.75]
    layer_indices = [max(0, min(n_layers - 1, round(n_layers * f) - 1)) for f in fracs]
    layer_pct_labels = ["25%", "50%", "75%"]

    print(f"n_layers={n_layers}, representative layers: "
          + ", ".join(f"{pct}→layer {idx}" for pct, idx in zip(layer_pct_labels, layer_indices)))
    print(f"k values to scan: {k_values}")
    print(f"Fitting PCA with max_k={max_k} per layer, then slicing for each k.\n")

    rows = []

    for layer_idx, layer_pct in zip(layer_indices, layer_pct_labels):
        layer_acts = activations[:, layer_idx, :]

        # Fit PCA once at max_k for this layer
        pca = PCA(n_components=max_k, random_state=42)
        acts_max = pca.fit_transform(layer_acts)           # (n_samples, max_k)
        cumvar = np.cumsum(pca.explained_variance_ratio_)  # cumulative variance per component

        print(f"Layer {layer_idx} ({layer_pct}):")

        for k in k_values:
            acts_k = acts_max[:, :k]                       # slice first k components
            var_explained = float(cumvar[k - 1])

            t0 = time.time()
            result = train_linear_probe(acts_k, labels)
            train_time = time.time() - t0

            f1_val   = result["f1_macro_val"]
            f1_train = result["f1_macro_train"]
            f1_gap   = f1_train - f1_val

            print(f"  k={k:4d} | var={var_explained:.3f} | "
                  f"val_F1={f1_val:.3f} | train_F1={f1_train:.3f} | "
                  f"gap={f1_gap:.3f} | time={train_time:.1f}s")

            rows.append({
                "layer_idx":        layer_idx,
                "layer_pct":        layer_pct,
                "k":                k,
                "variance_explained": var_explained,
                "f1_macro_val":     f1_val,
                "f1_macro_train":   f1_train,
                "f1_gap":           f1_gap,
                "train_time_sec":   round(train_time, 2),
            })

        print()

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df


def build_probe_dataset(
    tqa_full,
    mmlu_full,
    scenario_resp_df,
    output_path: Path,
    rules=None,
    include_social: bool = True,
) -> pd.DataFrame:
    """
    Build and save a probe dataset from judge results and (optionally) scenario responses.

    Skips if output_path already exists.

    Factual rows: filter_factual applied to tqa_full and mmlu_full with `rules`
    (defaults to the strictest 3-class thresholds, FACTUAL_RULES_DEFAULT).
    Social rows (only when include_social): honest → truth, deceptive → deception;
    system_prompt from scenario columns.

    Parameters
    ----------
    tqa_full, mmlu_full : full judge DataFrames (with 'config' and 'votes_correct')
    scenario_resp_df    : wide-format scenario responses; may be None when
                          include_social is False (e.g. supplement-config runs)
    output_path         : where to save the probe dataset CSV
    rules               : (config, votes_correct, label) triples passed to filter_factual
    include_social      : whether to append the social honest/deceptive rows

    Returns
    -------
    pd.DataFrame with columns: question, response, label, system_prompt, domain, correct_answer
    """
    output_path = Path(output_path)

    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"[skip] Loaded probe_dataset ({len(df)} rows): {output_path.name}")
        return df

    tqa_factual  = filter_factual(tqa_full,  "factual", rules=rules)
    mmlu_factual = filter_factual(mmlu_full, "factual", rules=rules)
    parts = [tqa_factual, mmlu_factual]

    if include_social:
        honest_rows = scenario_resp_df[["question", "honest_response", "honest_scenario"]].copy()
        honest_rows.columns = ["question", "response", "system_prompt"]
        honest_rows["label"] = "truth"

        deceptive_rows = scenario_resp_df[["question", "deceptive_response", "deceptive_scenario"]].copy()
        deceptive_rows.columns = ["question", "response", "system_prompt"]
        deceptive_rows["label"] = "deception"

        social = pd.concat([honest_rows, deceptive_rows], ignore_index=True)
        social["domain"] = "social"
        social["correct_answer"] = ""
        parts.append(social)

    probe_dataset = pd.concat(parts, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe_dataset.to_csv(output_path, index=False)
    print(f"Saved probe_dataset ({len(probe_dataset)} rows) → {output_path.name}")
    print(f"\nLabel distribution:\n{probe_dataset['label'].value_counts().to_string()}")
    print(f"\nDomain distribution:\n{probe_dataset['domain'].value_counts().to_string()}")
    return probe_dataset


def prepare_gemma4_thinking_dataset(
    dataset_path: str | Path,
    output_path: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Prepend the Gemma 4 thinking-mode trigger token to every prompt in the
    scenario dataset.

    Gemma 4 requires <|think|> at the start of the system prompt to enable
    extended thinking.  The base deception_dataset.csv does not include this
    token, so this function produces a ready-to-use variant.

    Parameters
    ----------
    dataset_path : path to deception_dataset.csv (or any CSV with a 'prompt' column)
    output_path  : where to save the result; defaults to
                   <dataset_path parent>/deception_dataset_gemma4_thinking.csv

    Returns
    -------
    pd.DataFrame — copy of the dataset with <|think|> prepended to every prompt
    """
    dataset_path = Path(dataset_path)
    if output_path is None:
        output_path = dataset_path.parent / "deception_dataset_gemma4_thinking.csv"
    output_path = Path(output_path)

    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"[skip] Loaded gemma4 thinking dataset ({len(df)} rows): {output_path.name}")
        return df

    df = pd.read_csv(dataset_path)
    out = df.copy()
    out["prompt"] = "<|think|>" + out["prompt"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved gemma4 thinking dataset ({len(out)} rows) → {output_path.name}")
    return out


def split_thinking_responses(
    df: pd.DataFrame,
    response_col: str = "response",
    save_path: "Path | str | None" = None,
) -> pd.DataFrame:
    """
    Split thinking-mode responses into thinking and answer parts.

    Detects and extracts thinking blocks from two formats:
      - Qwen3 / DeepSeek-R1 : <think>...</think>
      - Gemma 4             : <|channel>thought\\n...<channel|>

    Adds two columns to the returned DataFrame:
      response_thinking : content inside the thinking block (empty string if none)
      response_answer   : content after the thinking block (original response if no block)

    Rows without a thinking block are passed through with response_thinking='' and
    response_answer equal to the original response, so downstream code can always
    use response_answer as a unified column regardless of model.

    Parameters
    ----------
    df           : DataFrame containing raw model responses (thinking tags preserved)
    response_col : name of the column with raw responses (default: 'response')
    save_path    : if given, saves the result CSV to this path

    Returns
    -------
    pd.DataFrame — copy of df with response_thinking and response_answer added
    """
    _GEMMA_RE = re.compile(r"<\|channel>thought\n(.*?)<channel\|>(.*)", re.DOTALL)

    def _parse(text: str) -> tuple:
        if not isinstance(text, str):
            return ("", str(text))
        text = text.strip()
        # Qwen3/DeepSeek-R1: split on </think> (opening <think> may be absent)
        if "</think>" in text:
            thinking, answer = text.split("</think>", 1)
            if thinking.startswith("<think>"):
                thinking = thinking[len("<think>"):]
            return (thinking.strip(), answer.strip())
        # Gemma 4 format
        m = _GEMMA_RE.match(text)
        if m:
            return (m.group(1).strip(), m.group(2).strip())
        return ("", text)

    parsed = df[response_col].map(_parse)
    out = df.copy()
    out["response_thinking"] = parsed.map(lambda x: x[0])
    out["response_answer"]   = parsed.map(lambda x: x[1])

    has_thinking = (out["response_thinking"] != "").sum()
    print(f"Rows with thinking blocks : {has_thinking} / {len(out)}")
    print(f"Rows without thinking     : {len(out) - has_thinking} / {len(out)}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)
        print(f"Saved → {save_path.name}")

    return out


# Default factual (config, votes_correct, label) rules for the main 3-class experiment.
# Override via the `rules` argument — e.g. SUPPLEMENT_LABEL_RULES for the supplement configs.
FACTUAL_RULES_DEFAULT = [
    ("A", 6, "truth"),
    ("B", 0, "honest_mistake"),
    ("C", 0, "deception"),
]


def filter_factual(df, domain: str, rules=None) -> pd.DataFrame:
    """
    Filter a full judge DataFrame into probe-ready rows using (config, votes_correct) rules.

    Each rule is a (config, votes_correct, label) triple: rows whose 'config' AND
    'votes_correct' match are assigned that label. Rows matching no rule are discarded.

    The default rules (FACTUAL_RULES_DEFAULT) reproduce the strictest 3-class setting:
        Config A + votes_correct == 6 → truth
        Config B + votes_correct == 0 → honest_mistake
        Config C + votes_correct == 0 → deception

    Parameters
    ----------
    df     : full judge DataFrame with 'config' and 'votes_correct' columns
    domain : value written to the 'domain' column (e.g. 'factual')
    rules  : list of (config, votes_correct, label); defaults to FACTUAL_RULES_DEFAULT

    Returns
    -------
    pd.DataFrame with columns: question, response, label, system_prompt, domain, correct_answer
    """
    if rules is None:
        rules = FACTUAL_RULES_DEFAULT

    parts = []
    for config, votes_correct, label in rules:
        sub = df[(df["config"] == config) & (df["votes_correct"] == votes_correct)].copy()
        sub["label"] = label
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    out["domain"] = domain
    return out[["question", "response", "label", "system_prompt", "domain", "correct_answer"]]


def save_results_csv(results: list[dict], path: str | Path) -> pd.DataFrame:
    """
    Convert a list of per-layer probe result dicts to a DataFrame and save as CSV.

    Confusion matrix entries are flattened into columns named:
        cm_{true_class}_{pred_class}       (counts)
        cm_norm_{true_class}_{pred_class}  (row-normalized)

    All other scalar/dict fields are included as columns.

    Parameters
    ----------
    results : list of dicts from probe_all_layers or probe_all_layers_cascaded
    path    : output CSV path

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for r in results:
        row = {"layer": r["layer"], "f1_macro": r["f1_macro"]}

        # Per-class F1
        for cls, val in r["f1_per_class"].items():
            row[f"f1_{cls}"] = val

        # Confusion matrix (counts)
        classes = r["classes"]
        cm = r["confusion_matrix"]
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                row[f"cm_{true_cls}_{pred_cls}"] = int(cm[i, j])

        # Confusion matrix (normalized)
        cm_norm = r["confusion_matrix_norm"]
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                row[f"cm_norm_{true_cls}_{pred_cls}"] = float(cm_norm[i, j])

        # Cascaded-only fields
        if "stage1_f1" in r:
            for k, v in r["stage1_f1"].items():
                row[f"stage1_f1_{k}"] = v
        if "stage2_f1" in r:
            for k, v in r["stage2_f1"].items():
                row[f"stage2_f1_{k}"] = v
        if "stage2_auroc" in r:
            row["stage2_auroc"] = r["stage2_auroc"]

        rows.append(row)

    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def run_pca_reduction(
    activations: np.ndarray,
    n_components: int,
    acts_path: Path,
    components_path: Path,
    variance_path: Path,
    hf_repo: str = "",
    hf_token: str = "",
) -> np.ndarray:
    """
    Apply PCA independently per layer and save results.

    Priority:
    1. Load from local acts_path / components_path if both exist.
    2. Download from HuggingFace Hub (hf_repo) if local files missing.
    3. Compute from scratch using the activations array.

    Parameters
    ----------
    activations     : (n_samples, n_layers, hidden_dim)
    n_components    : number of PCA components (k)
    acts_path       : path to save reduced activations .npy
    components_path : path to save PCA components .npy
    variance_path   : path to save per-layer explained variance CSV
    hf_repo         : HuggingFace Hub repo ID (optional)
    hf_token        : HuggingFace token (optional)

    Returns
    -------
    np.ndarray : reduced activations (n_samples, n_layers, n_components)
    """
    acts_path       = Path(acts_path)
    components_path = Path(components_path)
    variance_path   = Path(variance_path)

    # ── 1. Local ──────────────────────────────────────────────────────────
    if acts_path.exists() and components_path.exists():
        acts_reduced = np.load(acts_path)
        print(f"[skip] Loaded reduced activations {acts_reduced.shape}: {acts_path.name}")
        return acts_reduced

    # ── 2. HuggingFace Hub ────────────────────────────────────────────────
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            try:
                from huggingface_hub.utils import enable_progress_bars
                enable_progress_bars()
            except ImportError:
                pass
            print(f"Local files not found. Downloading from {hf_repo} ...")
            # HF repo mirrors the local outputs/ tree — request the full repo-relative path.
            for path in [acts_path, components_path]:
                hf_hub_download(
                    repo_id=hf_repo, filename=path.as_posix(),
                    repo_type="dataset", token=hf_token,
                    local_dir=".",
                )
            acts_reduced = np.load(acts_path)
            print(f"[HF] Loaded reduced activations {acts_reduced.shape}: {acts_path.name}")
            return acts_reduced
        except Exception as e:
            print(f"Download failed ({e})")

    # ── 3. Compute from scratch ───────────────────────────────────────────
    print(f"Running PCA ({n_components} components) across {activations.shape[1]} layers ...")
    pca_result    = reduce_activations_pca(activations, n_components=n_components)
    acts_reduced  = pca_result["activations"]
    components    = pca_result["components"]
    explained_var = pca_result["explained_var"]

    np.save(acts_path,       acts_reduced)
    np.save(components_path, components)
    pd.DataFrame({
        "layer": range(len(explained_var)),
        "explained_var": explained_var,
    }).to_csv(variance_path, index=False)

    print(f"Saved {acts_path.name}       {acts_reduced.shape}")
    print(f"Saved {components_path.name} {components.shape}")
    print(f"Saved {variance_path.name}")
    print(f"Explained variance — mean: {explained_var.mean():.3f}, "
          f"min: {explained_var.min():.3f}, max: {explained_var.max():.3f}")
    return acts_reduced
