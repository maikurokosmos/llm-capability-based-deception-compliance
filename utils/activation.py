import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm


def extract_activations(
    question: str,
    response: str,
    system_prompt: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 8192,
) -> np.ndarray:
    """
    Run a single forward pass on (prompt + response) and extract the hidden state
    at the last token position for every transformer layer.

    `output_hidden_states=True` keeps every layer's hidden state for the whole
    sequence, so a single pathological row — e.g. a degenerate greedy-decoding
    repetition running to tens of thousands of tokens — can exhaust VRAM on any card.
    `max_length` caps that. Truncation is LEFT-side so the genuine final token (the
    position we read) is preserved; legitimate rows sit far below the cap and are
    never touched.

    Returns
    -------
    np.ndarray of shape (n_layers, hidden_dim)
        Index 0 = layer 1, ..., index n_layers-1 = last layer.
        (Embedding layer is excluded.)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = prompt_text + response
    tokenizer.truncation_side = "left"
    input_ids = tokenizer(
        full_text, return_tensors="pt",
        truncation=True, max_length=max_length,
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # outputs.hidden_states: tuple of (n_layers + 1) tensors, shape (1, seq_len, hidden_dim)
    # Index 0 is the embedding layer — skip it, keep layers 1..n_layers
    hidden_states = outputs.hidden_states[1:]  # (n_layers,) each: (1, seq_len, hidden_dim)

    # Take the last token position from each layer → (n_layers, hidden_dim)
    activations = torch.stack([hs[0, -1, :] for hs in hidden_states])
    return activations.cpu().float().numpy()


def count_prompt_tokens(question, response, system_prompt, tokenizer) -> int:
    """Token length of the exact (prompt + response) string extract_activations() feeds
    the model — used to filter degenerate ultra-long rows before extraction."""
    messages = [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user",   "content": str(question)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return len(tokenizer(prompt_text + str(response)).input_ids)


def drop_overlong_rows(probe_dataset, tokenizer, max_length: int, response_col: str = "response"):
    """Drop rows whose tokenized (prompt + response) exceeds max_length.

    Targets degenerate generations — e.g. a greedy-decoding repetition loop running to
    tens of thousands of tokens — which would OOM extraction (output_hidden_states keeps
    every layer's hidden state for the whole sequence) and are junk data anyway. Order is
    preserved; returns a reindexed copy and prints what was removed.
    """
    n_tok = np.array([
        count_prompt_tokens(r.question, getattr(r, response_col), r.system_prompt, tokenizer)
        for r in probe_dataset.itertuples()
    ])
    keep = n_tok <= max_length
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"[filter] dropped {n_drop} row(s) > {max_length} tokens "
              f"(longest was {int(n_tok.max())}); kept {int(keep.sum())}/{len(probe_dataset)}")
    else:
        print(f"[filter] no rows exceed {max_length} tokens; kept all {len(probe_dataset)}")
    return probe_dataset[keep].reset_index(drop=True)


LABEL_MAP = {
    "truth": 0,
    "honest_mistake": 1,
    "deception": 2,
    # Supplement configs (additional_experiment.ipynb). Append-only so existing
    # labels.npy files keep decoding to the original three classes.
    "natural_deception": 3,
    "capable_failed": 4,
    "deception_rejection": 5,
}


def run_extract_activations(
    probe_dataset,
    model,
    tokenizer,
    device: str,
    activations_path: Path,
    labels_path: Path,
    checkpoint_path: Path,
    hf_repo: str,
    hf_token: str,
    checkpoint_every: int = 50,
    max_length: int = 8192,
):
    """
    Load or extract activations for all rows in probe_dataset.

    Priority:
    1. Load from local activations_path / labels_path if both exist.
    2. Download from HuggingFace Hub (hf_repo) if local files missing.
    3. Extract from scratch using the model, with checkpoint/resume support.

    Returns
    -------
    activations_arr : np.ndarray (n_samples, n_layers, hidden_dim)
    labels_arr      : np.ndarray (n_samples,) — integer encoded
    """
    activations_path = Path(activations_path)
    labels_path      = Path(labels_path)
    checkpoint_path  = Path(checkpoint_path)

    def _load_with_progress(path, desc):
        size = path.stat().st_size
        with tqdm.wrapattr(open(path, "rb"), "read", total=size, desc=desc) as f:
            return np.load(f)

    # ── 1. Local ──────────────────────────────────────────────────────────
    if activations_path.exists() and labels_path.exists():
        activations_arr = _load_with_progress(activations_path, "Loading activations")
        labels_arr      = _load_with_progress(labels_path,      "Loading labels     ")
        print(f"[local] activations {activations_arr.shape}, labels {labels_arr.shape}")
        return activations_arr, labels_arr

    # ── 2. HuggingFace Hub ────────────────────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download
        try:
            from huggingface_hub.utils import enable_progress_bars
            enable_progress_bars()
        except ImportError:
            pass
        print(f"Local files not found. Downloading from {hf_repo} ...")
        # The HF repo mirrors the local outputs/ tree (uploaded via
        # upload_folder(path_in_repo="outputs")), so request the full repo-relative path
        # and restore it under the repo root. Using the full path also means a run with
        # distinct names (e.g. activations_new_config.npy) never pulls the 3-class file by
        # mistake — a missing file 404s and falls through to extraction.
        for path in [activations_path, labels_path]:
            hf_hub_download(
                repo_id=hf_repo, filename=path.as_posix(),
                repo_type="dataset", token=hf_token,
                local_dir=".",
            )
        activations_arr = _load_with_progress(activations_path, "Loading activations")
        labels_arr      = _load_with_progress(labels_path,      "Loading labels     ")
        print(f"[HF] activations {activations_arr.shape}, labels {labels_arr.shape}")
        return activations_arr, labels_arr

    except Exception as e:
        print(f"Download failed ({type(e).__name__}: {e}). Running extraction ...")

    # ── 3. Extract ────────────────────────────────────────────────────────
    if checkpoint_path.exists():
        ckpt = np.load(checkpoint_path)
        all_activations = list(ckpt["activations"])
        all_labels      = list(ckpt["labels"])
        start_idx = len(all_activations)
        print(f"Resuming from checkpoint: {start_idx}/{len(probe_dataset)} done")
    else:
        all_activations, all_labels, start_idx = [], [], 0
        print(f"Starting fresh: {len(probe_dataset)} samples")

    for i, row in enumerate(tqdm(
        probe_dataset.iloc[start_idx:].itertuples(),
        total=len(probe_dataset) - start_idx,
        desc="Extracting activations",
    )):
        all_activations.append(extract_activations(
            question=row.question,
            response=row.response,
            system_prompt=row.system_prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        ))
        all_labels.append(LABEL_MAP[row.label])

        if (start_idx + i + 1) % checkpoint_every == 0:
            np.savez(checkpoint_path,
                     activations=np.array(all_activations),
                     labels=np.array(all_labels))

    activations_arr = np.array(all_activations)
    labels_arr      = np.array(all_labels)
    np.save(activations_path, activations_arr)
    np.save(labels_path,      labels_arr)
    print(f"Extracted and saved: activations {activations_arr.shape}")
    return activations_arr, labels_arr
