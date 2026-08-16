import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from tqdm.auto import tqdm


def train_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    max_iter: int = 200,
    random_state: int = 42,
    C: float = 1.0,
) -> dict:
    """
    Train a logistic regression probe on activations from a single layer
    using stratified k-fold cross-validation.

    Parameters
    ----------
    activations : (n_samples, hidden_dim)
    labels      : (n_samples,) — string labels "truth"/"honest_mistake"/"deception"

    Returns
    -------
    dict with keys:
        f1_per_class          : dict {class_name: mean_f1_across_folds}
        f1_macro              : float
        confusion_matrix      : np.ndarray (n_classes, n_classes), accumulated counts
        confusion_matrix_norm : np.ndarray (n_classes, n_classes), row-normalized (recall view)
        classes               : list of class names in matrix row/col order
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = sorted(set(labels))

    fold_f1s_val   = {c: [] for c in classes}
    fold_f1s_train = {c: [] for c in classes}
    cm_accum = np.zeros((len(classes), len(classes)), dtype=int)

    for train_idx, val_idx in skf.split(activations, labels):
        X_train, X_val = activations[train_idx], activations[val_idx]
        y_train, y_val = labels[train_idx],      labels[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        clf = LogisticRegression(solver="saga", max_iter=max_iter, random_state=random_state, C=C)
        clf.fit(X_train, y_train)

        y_pred       = clf.predict(X_val)
        y_train_pred = clf.predict(X_train)

        f1s_val = f1_score(y_val, y_pred, labels=classes, average=None)
        for cls, score in zip(classes, f1s_val):
            fold_f1s_val[cls].append(score)

        f1s_train = f1_score(y_train, y_train_pred, labels=classes, average=None)
        for cls, score in zip(classes, f1s_train):
            fold_f1s_train[cls].append(score)

        cm_accum += confusion_matrix(y_val, y_pred, labels=classes)

    f1_per_class       = {cls: float(np.mean(scores)) for cls, scores in fold_f1s_val.items()}
    f1_per_class_train = {cls: float(np.mean(scores)) for cls, scores in fold_f1s_train.items()}
    f1_macro_val   = float(np.mean(list(f1_per_class.values())))
    f1_macro_train = float(np.mean(list(f1_per_class_train.values())))

    row_sums = cm_accum.sum(axis=1, keepdims=True)
    cm_norm = cm_accum.astype(float) / np.where(row_sums == 0, 1, row_sums)

    return {
        "f1_per_class":          f1_per_class,
        "f1_per_class_train":    f1_per_class_train,
        "f1_macro":              f1_macro_val,
        "f1_macro_val":          f1_macro_val,
        "f1_macro_train":        f1_macro_train,
        "confusion_matrix":      cm_accum,
        "confusion_matrix_norm": cm_norm,
        "classes":               classes,
    }


def _save_probe_csv(results: list[dict], path: Path) -> pd.DataFrame:
    """Flatten per-layer probe result dicts to a DataFrame and save as CSV."""
    rows = []
    for r in results:
        row = {"layer": r["layer"], "f1_macro": r["f1_macro"]}
        for cls, val in r["f1_per_class"].items():
            row[f"f1_{cls}"] = val
        classes = r["classes"]
        cm      = r["confusion_matrix"]
        cm_norm = r["confusion_matrix_norm"]
        for i, tc in enumerate(classes):
            for j, pc in enumerate(classes):
                row[f"cm_{tc}_{pc}"]      = int(cm[i, j])
                row[f"cm_norm_{tc}_{pc}"] = float(cm_norm[i, j])
        for key in ("auroc", "n_samples", "stage2_auroc"):
            if key in r:
                row[key] = r[key]
        for prefix in ("stage1_f1", "stage2_f1"):
            if prefix in r:
                for k, v in r[prefix].items():
                    row[f"{prefix}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def probe_all_layers(
    activations: np.ndarray,
    labels: np.ndarray,
    output_path: Path = None,
    checkpoint_path: Path = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run train_linear_probe for every layer with skip/checkpoint/save support.

    Parameters
    ----------
    activations     : (n_samples, n_layers, hidden_dim)
    output_path     : if provided, skip if CSV exists; save results on completion
    checkpoint_path : if provided, resume from pickle checkpoint

    Returns
    -------
    pd.DataFrame — one row per layer
    """
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists():
            df = pd.read_csv(output_path)
            print(f"[skip] {output_path.name} already exists ({len(df)} rows)")
            return df

    n_layers = activations.shape[1]

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        done = {r["layer"] for r in results}
        print(f"[checkpoint] Resuming from layer {max(done)+1} ({len(done)}/{n_layers} done)")
    else:
        results, done = [], set()

    for layer_idx in tqdm(range(n_layers), desc="3-way LR probe (layers)"):
        if layer_idx in done:
            continue
        result = train_linear_probe(activations[:, layer_idx, :], labels, **kwargs)
        result["layer"] = layer_idx
        results.append(result)
        results.sort(key=lambda r: r["layer"])
        if checkpoint_path is not None:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

    if output_path is not None:
        df = _save_probe_csv(results, output_path)
        print(f"Saved {output_path.name} ({len(df)} rows)")
        return df
    return results


def train_binary_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    max_iter: int = 200,
    random_state: int = 42,
    C: float = 1.0,
    pos_label: str = "deception",
    drop_labels=("honest_mistake",),
) -> dict:
    """
    Train a binary logistic regression probe on exactly two classes.

    Defaults reproduce the original truth-vs-deception probe: honest_mistake rows are
    dropped and 'deception' is the positive class for AUROC. For arbitrary pairs (e.g. the
    supplement-config binaries) pass a pre-subset two-class set with drop_labels=() and
    pos_label set to the class of interest. AUROC is invariant to the pos_label choice — it
    only affects which class is labelled positive.

    Parameters
    ----------
    activations : (n_samples, hidden_dim)
    labels      : (n_samples,) — string labels
    C           : inverse regularization strength (use C=0.1 to match Goldowsky et al. λ=10)
    pos_label   : class treated as positive for AUROC (default "deception")
    drop_labels : labels removed before training (default drops honest_mistake to preserve
                  the original truth-vs-deception behaviour; pass () for explicit pairs)

    Returns
    -------
    dict with keys:
        auroc                : float — mean AUROC across folds (primary metric)
        f1_macro             : float — mean macro F1 across folds
        f1_per_class         : dict {class_name: mean_f1_across_folds}
        confusion_matrix     : np.ndarray (2, 2), accumulated counts
        confusion_matrix_norm: np.ndarray (2, 2), row-normalized
        classes              : sorted list of the two class names
        n_samples            : int — number of samples after filtering
    """
    labels = np.array(labels)
    if drop_labels:
        keep = ~np.isin(labels, list(drop_labels))
        activations = activations[keep]
        labels      = labels[keep]
    n_samples = len(labels)

    classes = sorted(set(labels))
    pos_idx = classes.index(pos_label)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_f1s  = {c: [] for c in classes}
    fold_aurocs = []
    cm_accum  = np.zeros((len(classes), len(classes)), dtype=int)

    for train_idx, val_idx in skf.split(activations, labels):
        X_train, X_val = activations[train_idx], activations[val_idx]
        y_train, y_val = labels[train_idx],      labels[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        clf = LogisticRegression(solver="saga", max_iter=max_iter, random_state=random_state, C=C)
        clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:, pos_idx]

        f1s = f1_score(y_val, y_pred, labels=classes, average=None)
        for cls, score in zip(classes, f1s):
            fold_f1s[cls].append(score)

        fold_aurocs.append(roc_auc_score((y_val == pos_label).astype(int), y_proba))
        cm_accum += confusion_matrix(y_val, y_pred, labels=classes)

    f1_per_class = {cls: float(np.mean(scores)) for cls, scores in fold_f1s.items()}
    f1_macro     = float(np.mean(list(f1_per_class.values())))
    auroc        = float(np.mean(fold_aurocs))

    row_sums = cm_accum.sum(axis=1, keepdims=True)
    cm_norm  = cm_accum.astype(float) / np.where(row_sums == 0, 1, row_sums)

    return {
        "auroc":               auroc,
        "f1_macro":            f1_macro,
        "f1_per_class":        f1_per_class,
        "confusion_matrix":    cm_accum,
        "confusion_matrix_norm": cm_norm,
        "classes":             classes,
        "n_samples":           n_samples,
    }


def probe_all_layers_binary(
    activations: np.ndarray,
    labels: np.ndarray,
    output_path: Path = None,
    checkpoint_path: Path = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run train_binary_probe for every layer with skip/checkpoint/save support.

    Parameters
    ----------
    activations     : (n_samples, n_layers, hidden_dim)
    output_path     : if provided, skip if CSV exists; save results on completion
    checkpoint_path : if provided, resume from pickle checkpoint

    Returns
    -------
    pd.DataFrame — one row per layer
    """
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists():
            df = pd.read_csv(output_path)
            print(f"[skip] {output_path.name} already exists ({len(df)} rows)")
            return df

    n_layers = activations.shape[1]

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        done = {r["layer"] for r in results}
        print(f"[checkpoint] Resuming from layer {max(done)+1} ({len(done)}/{n_layers} done)")
    else:
        results, done = [], set()

    for layer_idx in tqdm(range(n_layers), desc="binary probe (layers)"):
        if layer_idx in done:
            continue
        result = train_binary_probe(activations[:, layer_idx, :], labels, **kwargs)
        result["layer"] = layer_idx
        results.append(result)
        results.sort(key=lambda r: r["layer"])
        if checkpoint_path is not None:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

    if output_path is not None:
        df = _save_probe_csv(results, output_path)
        print(f"Saved {output_path.name} ({len(df)} rows)")
        return df
    return results


def train_cascaded_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    max_iter: int = 200,
    random_state: int = 42,
    C: float = 1.0,
) -> dict:
    """
    Train a two-stage cascaded probe on activations from a single layer
    using stratified k-fold cross-validation.

    Stage 1: truth vs non_truth (honest_mistake + deception)
    Stage 2: honest_mistake vs deception, on non_truth samples only

    Parameters
    ----------
    activations : (n_samples, hidden_dim)
    labels      : (n_samples,) — string labels "truth"/"honest_mistake"/"deception"

    Returns
    -------
    dict with keys:
        f1_per_class          : dict {class_name: overall 3-way f1}
        f1_macro              : float, overall 3-way macro F1
        confusion_matrix      : np.ndarray (3,3), accumulated counts
        confusion_matrix_norm : np.ndarray (3,3), row-normalized
        classes               : list of class names in matrix row/col order
        stage1_f1             : dict {"truth": float, "non_truth": float}
        stage2_f1             : dict {"honest_mistake": float, "deception": float}
        stage2_auroc          : float, AUROC for deception (positive class) in stage 2
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = sorted(set(labels))  # ['deception', 'honest_mistake', 'truth']

    cm_accum = np.zeros((len(classes), len(classes)), dtype=int)

    s1_fold_f1s = {"truth": [], "non_truth": []}
    s2_fold_f1s = {"honest_mistake": [], "deception": []}
    s2_fold_aurocs = []

    labels_arr = np.array(labels)

    for train_idx, val_idx in skf.split(activations, labels_arr):
        X_train, X_val = activations[train_idx], activations[val_idx]
        y_train, y_val = labels_arr[train_idx],  labels_arr[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # ── Stage 1: truth vs non_truth ───────────────────────────────────
        s1_train = np.where(y_train == "truth", "truth", "non_truth")
        s1_val   = np.where(y_val   == "truth", "truth", "non_truth")

        clf1 = LogisticRegression(solver="saga", max_iter=max_iter, random_state=random_state, C=C)
        clf1.fit(X_train, s1_train)
        s1_pred = clf1.predict(X_val)

        s1_f1s = f1_score(s1_val, s1_pred, labels=["truth", "non_truth"], average=None)
        s1_fold_f1s["truth"].append(s1_f1s[0])
        s1_fold_f1s["non_truth"].append(s1_f1s[1])

        # ── Stage 2: honest_mistake vs deception (non_truth samples only) ─
        s2_mask_train = y_train != "truth"
        s2_mask_val   = y_val   != "truth"

        s2_classes = ["deception", "honest_mistake"]
        if s2_mask_train.sum() >= 2 and len(set(y_train[s2_mask_train])) == 2:
            clf2 = LogisticRegression(solver="saga", max_iter=max_iter, random_state=random_state, C=C)
            clf2.fit(X_train[s2_mask_train], y_train[s2_mask_train])

            # Stage 2 metrics: evaluate on TRUE non-truth val samples (oracle signal)
            s2_pred_oracle  = clf2.predict(X_val[s2_mask_val])
            s2_proba_oracle = clf2.predict_proba(X_val[s2_mask_val])
            dec_idx         = list(clf2.classes_).index("deception")

            s2_f1s = f1_score(
                y_val[s2_mask_val], s2_pred_oracle,
                labels=s2_classes, average=None,
            )
            s2_fold_f1s["deception"].append(s2_f1s[0])
            s2_fold_f1s["honest_mistake"].append(s2_f1s[1])

            auroc = roc_auc_score(
                (y_val[s2_mask_val] == "deception").astype(int),
                s2_proba_oracle[:, dec_idx],
            )
            s2_fold_aurocs.append(auroc)

            # Real cascaded routing: use stage 1 prediction to route to stage 2
            y_combined = np.empty(len(y_val), dtype=object)
            s1_truth_mask    = s1_pred == "truth"
            s1_nontruth_mask = ~s1_truth_mask
            y_combined[s1_truth_mask]    = "truth"
            y_combined[s1_nontruth_mask] = clf2.predict(X_val[s1_nontruth_mask])
            cm_accum += confusion_matrix(y_val, y_combined, labels=classes)

        else:
            s2_fold_f1s["deception"].append(float("nan"))
            s2_fold_f1s["honest_mistake"].append(float("nan"))
            s2_fold_aurocs.append(float("nan"))
            y_combined = np.where(s1_pred == "truth", "truth", "honest_mistake")
            cm_accum += confusion_matrix(y_val, y_combined, labels=classes)

    # ── Aggregate metrics ─────────────────────────────────────────────────
    row_sums = cm_accum.sum(axis=1, keepdims=True)
    cm_norm  = cm_accum.astype(float) / np.where(row_sums == 0, 1, row_sums)

    # Overall 3-way F1 from accumulated confusion matrix
    f1_per_class = {}
    for i, cls in enumerate(classes):
        tp = cm_accum[i, i]
        fp = cm_accum[:, i].sum() - tp
        fn = cm_accum[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1_per_class[cls] = float(2 * tp / denom) if denom > 0 else 0.0
    f1_macro = float(np.mean(list(f1_per_class.values())))

    return {
        "f1_per_class":          f1_per_class,
        "f1_macro":              f1_macro,
        "confusion_matrix":      cm_accum,
        "confusion_matrix_norm": cm_norm,
        "classes":               classes,
        "stage1_f1":             {k: float(np.nanmean(v)) for k, v in s1_fold_f1s.items()},
        "stage2_f1":             {k: float(np.nanmean(v)) for k, v in s2_fold_f1s.items()},
        "stage2_auroc":          float(np.nanmean(s2_fold_aurocs)),
    }


def probe_all_layers_cascaded(
    activations: np.ndarray,
    labels: np.ndarray,
    output_path: Path = None,
    checkpoint_path: Path = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run train_cascaded_probe for every layer with skip/checkpoint/save support.

    Parameters
    ----------
    activations     : (n_samples, n_layers, hidden_dim)
    output_path     : if provided, skip if CSV exists; save results on completion
    checkpoint_path : if provided, resume from pickle checkpoint

    Returns
    -------
    pd.DataFrame — one row per layer
    """
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists():
            df = pd.read_csv(output_path)
            print(f"[skip] {output_path.name} already exists ({len(df)} rows)")
            return df

    n_layers = activations.shape[1]

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        done = {r["layer"] for r in results}
        print(f"[checkpoint] Resuming from layer {max(done)+1} ({len(done)}/{n_layers} done)")
    else:
        results, done = [], set()

    for layer_idx in tqdm(range(n_layers), desc="cascaded LR probe (layers)"):
        if layer_idx in done:
            continue
        result = train_cascaded_probe(activations[:, layer_idx, :], labels, **kwargs)
        result["layer"] = layer_idx
        results.append(result)
        results.sort(key=lambda r: r["layer"])
        if checkpoint_path is not None:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

    if output_path is not None:
        df = _save_probe_csv(results, output_path)
        print(f"Saved {output_path.name} ({len(df)} rows)")
        return df
    return results


def train_mlp_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    max_iter: int = 200,
    random_state: int = 42,
    hidden_layer_sizes: tuple = (256,),
) -> dict:
    """
    Train a 3-way MLP probe on activations from a single layer
    using stratified k-fold cross-validation.

    Parameters
    ----------
    activations : (n_samples, hidden_dim)
    labels      : (n_samples,) — string labels "truth"/"honest_mistake"/"deception"

    Returns
    -------
    dict with same keys as train_linear_probe
    """
    classes = sorted(set(labels))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_f1s_val   = {c: [] for c in classes}
    fold_f1s_train = {c: [] for c in classes}
    cm_accum       = np.zeros((len(classes), len(classes)), dtype=int)

    for train_idx, val_idx in skf.split(activations, labels):
        X_train, X_val = activations[train_idx], activations[val_idx]
        y_train, y_val = np.array(labels)[train_idx], np.array(labels)[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        y_pred_val   = clf.predict(X_val)
        y_pred_train = clf.predict(X_train)

        f1s_val   = f1_score(y_val,   y_pred_val,   labels=classes, average=None)
        f1s_train = f1_score(y_train, y_pred_train, labels=classes, average=None)
        for cls, v, t in zip(classes, f1s_val, f1s_train):
            fold_f1s_val[cls].append(v)
            fold_f1s_train[cls].append(t)

        cm_accum += confusion_matrix(y_val, y_pred_val, labels=classes)

    f1_per_class_val   = {cls: float(np.mean(scores)) for cls, scores in fold_f1s_val.items()}
    f1_per_class_train = {cls: float(np.mean(scores)) for cls, scores in fold_f1s_train.items()}
    f1_macro_val       = float(np.mean(list(f1_per_class_val.values())))
    f1_macro_train     = float(np.mean(list(f1_per_class_train.values())))

    row_sums = cm_accum.sum(axis=1, keepdims=True)
    cm_norm  = cm_accum.astype(float) / np.where(row_sums == 0, 1, row_sums)

    return {
        "f1_macro_val":      f1_macro_val,
        "f1_macro_train":    f1_macro_train,
        "f1_per_class_val":  f1_per_class_val,
        "f1_per_class_train": f1_per_class_train,
        "f1_macro":          f1_macro_val,
        "f1_per_class":      f1_per_class_val,
        "confusion_matrix":     cm_accum,
        "confusion_matrix_norm": cm_norm,
        "classes":           classes,
    }


def probe_all_layers_mlp(
    activations: np.ndarray,
    labels: np.ndarray,
    output_path: Path = None,
    checkpoint_path: Path = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run train_mlp_probe for every layer with skip/checkpoint/save support.

    Parameters
    ----------
    activations     : (n_samples, n_layers, hidden_dim)
    output_path     : if provided, skip if CSV exists; save results on completion
    checkpoint_path : if provided, resume from pickle checkpoint

    Returns
    -------
    pd.DataFrame — one row per layer
    """
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists():
            df = pd.read_csv(output_path)
            print(f"[skip] {output_path.name} already exists ({len(df)} rows)")
            return df

    n_layers = activations.shape[1]

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        done = {r["layer"] for r in results}
        print(f"[checkpoint] Resuming from layer {max(done)+1} ({len(done)}/{n_layers} done)")
    else:
        results, done = [], set()

    for layer_idx in tqdm(range(n_layers), desc="3-way MLP probe (layers)"):
        if layer_idx in done:
            continue
        result = train_mlp_probe(activations[:, layer_idx, :], labels, **kwargs)
        result["layer"] = layer_idx
        results.append(result)
        results.sort(key=lambda r: r["layer"])
        if checkpoint_path is not None:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

    if output_path is not None:
        df = _save_probe_csv(results, output_path)
        print(f"Saved {output_path.name} ({len(df)} rows)")
        return df
    return results


def train_cascaded_mlp_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    max_iter: int = 200,
    random_state: int = 42,
    hidden_layer_sizes: tuple = (256,),
) -> dict:
    """
    Train a two-stage cascaded MLP probe on activations from a single layer
    using stratified k-fold cross-validation.

    Stage 1: truth vs non_truth (honest_mistake + deception)
    Stage 2: honest_mistake vs deception, on non_truth samples only

    Returns
    -------
    dict with same keys as train_cascaded_probe
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = sorted(set(labels))  # ['deception', 'honest_mistake', 'truth']

    cm_accum = np.zeros((len(classes), len(classes)), dtype=int)

    s1_fold_f1s    = {"truth": [], "non_truth": []}
    s2_fold_f1s    = {"honest_mistake": [], "deception": []}
    s2_fold_aurocs = []

    labels_arr = np.array(labels)

    for train_idx, val_idx in skf.split(activations, labels_arr):
        X_train, X_val = activations[train_idx], activations[val_idx]
        y_train, y_val = labels_arr[train_idx],  labels_arr[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # ── Stage 1: truth vs non_truth ───────────────────────────────────
        s1_train = np.where(y_train == "truth", "truth", "non_truth")
        s1_val   = np.where(y_val   == "truth", "truth", "non_truth")

        clf1 = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu", solver="adam",
            max_iter=max_iter, random_state=random_state,
        )
        clf1.fit(X_train, s1_train)
        s1_pred = clf1.predict(X_val)

        s1_f1s = f1_score(s1_val, s1_pred, labels=["truth", "non_truth"], average=None)
        s1_fold_f1s["truth"].append(s1_f1s[0])
        s1_fold_f1s["non_truth"].append(s1_f1s[1])

        # ── Stage 2: honest_mistake vs deception (non_truth only) ─────────
        s2_mask_train = y_train != "truth"
        s2_mask_val   = y_val   != "truth"

        s2_classes = ["deception", "honest_mistake"]
        if s2_mask_train.sum() >= 2 and len(set(y_train[s2_mask_train])) == 2:
            clf2 = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu", solver="adam",
                max_iter=max_iter, random_state=random_state,
            )
            clf2.fit(X_train[s2_mask_train], y_train[s2_mask_train])

            s2_pred_oracle  = clf2.predict(X_val[s2_mask_val])
            s2_proba_oracle = clf2.predict_proba(X_val[s2_mask_val])
            dec_idx         = list(clf2.classes_).index("deception")

            s2_f1s = f1_score(
                y_val[s2_mask_val], s2_pred_oracle,
                labels=s2_classes, average=None,
            )
            s2_fold_f1s["deception"].append(s2_f1s[0])
            s2_fold_f1s["honest_mistake"].append(s2_f1s[1])
            s2_fold_aurocs.append(
                roc_auc_score(
                    (y_val[s2_mask_val] == "deception").astype(int),
                    s2_proba_oracle[:, dec_idx],
                )
            )

            # Real cascaded routing: use stage 1 prediction to route to stage 2
            y_pred_full = np.empty(len(y_val), dtype=object)
            s1_truth_mask    = s1_pred == "truth"
            s1_nontruth_mask = ~s1_truth_mask
            y_pred_full[s1_truth_mask]    = "truth"
            y_pred_full[s1_nontruth_mask] = clf2.predict(X_val[s1_nontruth_mask])
            cm_accum += confusion_matrix(y_val, y_pred_full, labels=classes)

    stage1_f1  = {cls: float(np.mean(scores)) for cls, scores in s1_fold_f1s.items()}
    stage2_f1  = {cls: float(np.mean(scores)) for cls, scores in s2_fold_f1s.items()}
    stage2_auroc = float(np.mean(s2_fold_aurocs)) if s2_fold_aurocs else float("nan")

    # Overall 3-way F1 from accumulated confusion matrix
    row_sums = cm_accum.sum(axis=1, keepdims=True)
    cm_norm  = cm_accum.astype(float) / np.where(row_sums == 0, 1, row_sums)

    per_class_f1 = {}
    for i, cls in enumerate(classes):
        tp = cm_accum[i, i]
        fp = cm_accum[:, i].sum() - tp
        fn = cm_accum[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        per_class_f1[cls] = float(2 * tp / denom) if denom > 0 else 0.0
    f1_macro = float(np.mean(list(per_class_f1.values())))

    return {
        "f1_macro":              f1_macro,
        "f1_per_class":          per_class_f1,
        "confusion_matrix":      cm_accum,
        "confusion_matrix_norm": cm_norm,
        "classes":               classes,
        "stage1_f1":             stage1_f1,
        "stage2_f1":             stage2_f1,
        "stage2_auroc":          stage2_auroc,
    }


def probe_all_layers_cascaded_mlp(
    activations: np.ndarray,
    labels: np.ndarray,
    output_path: Path = None,
    checkpoint_path: Path = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run train_cascaded_mlp_probe for every layer with skip/checkpoint/save support.

    Parameters
    ----------
    activations     : (n_samples, n_layers, hidden_dim)
    output_path     : if provided, skip if CSV exists; save results on completion
    checkpoint_path : if provided, resume from pickle checkpoint

    Returns
    -------
    pd.DataFrame — one row per layer
    """
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists():
            df = pd.read_csv(output_path)
            print(f"[skip] {output_path.name} already exists ({len(df)} rows)")
            return df

    n_layers = activations.shape[1]

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        done = {r["layer"] for r in results}
        print(f"[checkpoint] Resuming from layer {max(done)+1} ({len(done)}/{n_layers} done)")
    else:
        results, done = [], set()

    for layer_idx in tqdm(range(n_layers), desc="cascaded MLP probe (layers)"):
        if layer_idx in done:
            continue
        result = train_cascaded_mlp_probe(activations[:, layer_idx, :], labels, **kwargs)
        result["layer"] = layer_idx
        results.append(result)
        results.sort(key=lambda r: r["layer"])
        if checkpoint_path is not None:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

    if output_path is not None:
        df = _save_probe_csv(results, output_path)
        print(f"Saved {output_path.name} ({len(df)} rows)")
        return df
    return results
