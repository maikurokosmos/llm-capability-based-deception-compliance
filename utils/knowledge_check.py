import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm

_loss_fn = nn.CrossEntropyLoss(reduction="none")


def score_choice(
    question: str,
    choice: str,
    model,
    tokenizer,
    device: str,
) -> float:
    """
    Average NLL of `choice` tokens given `question` as context.
    Lower = model considers this choice more likely.
    """
    messages = [
        {"role": "system", "content": "Answer the following question concisely and factually."},
        {"role": "user",   "content": question},
    ]
    context_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = context_text + choice

    context_ids = tokenizer(context_text, return_tensors="pt").input_ids.to(device)
    full_ids    = tokenizer(full_text,    return_tensors="pt").input_ids.to(device)

    choice_len = full_ids.shape[-1] - context_ids.shape[-1]
    if choice_len <= 0:
        return float("inf")

    with torch.no_grad():
        logits = model(full_ids).logits[0]      # (seq_len, vocab)

    shift_logits = logits[:-1]                  # (seq_len-1, vocab)
    shift_labels = full_ids[0, 1:]              # (seq_len-1,)

    token_nll  = _loss_fn(shift_logits, shift_labels)
    choice_nll = token_nll[-choice_len:]
    return choice_nll.mean().item()


def knowledge_check_truthfulqa(
    item: dict,
    model,
    tokenizer,
    device: str,
) -> dict:
    """
    Knowledge check for TruthfulQA mc1_targets format.
    item must have keys: "question", "mc1_targets" (with "choices" and "labels").

    Returns:
        question, all_choices, all_scores, model_choice, correct_answer, passed
    """
    question = item["question"]
    choices  = item["mc1_targets"]["choices"]
    labels   = item["mc1_targets"]["labels"]
    correct  = choices[labels.index(1)]

    scores   = [score_choice(question, c, model, tokenizer, device) for c in choices]
    best_idx = int(np.argmin(scores))

    return {
        "question":       question,
        "all_choices":    choices,
        "all_scores":     scores,
        "model_choice":   choices[best_idx],
        "correct_answer": correct,
        "passed":         choices[best_idx] == correct,
    }


def run_knowledge_check_truthfulqa(
    dataset,
    model,
    tokenizer,
    device: str,
    output_path: Path,
    checkpoint_every: int = 50,
):
    """
    Run knowledge check on TruthfulQA MC dataset with checkpoint/resume support.

    Skips entirely if output_path already contains all rows.
    Resumes from partial checkpoint if output_path exists but is incomplete.

    Returns
    -------
    kc_df     : pd.DataFrame — full results
    passed_df : pd.DataFrame — rows where model answered correctly
    failed_df : pd.DataFrame — rows where model answered incorrectly
    """
    output_path = Path(output_path)
    total = len(dataset)

    if output_path.exists():
        kc_df = pd.read_csv(output_path)
        if len(kc_df) == total:
            print(f"[skip] Already complete ({total} rows): {output_path.name}")
            passed_df = kc_df[kc_df["passed"]].reset_index(drop=True)
            failed_df = kc_df[~kc_df["passed"]].reset_index(drop=True)
            return kc_df, passed_df, failed_df
        done_questions = set(kc_df["question"].tolist())
        remaining = [item for item in dataset if item["question"] not in done_questions]
        print(f"Resuming: {len(kc_df)} done, {len(remaining)} remaining")
    else:
        remaining = list(dataset)
        print(f"Starting fresh: {total} items")

    records = []
    for i, item in enumerate(tqdm(remaining, desc="TruthfulQA knowledge check")):
        records.append(knowledge_check_truthfulqa(item, model, tokenizer, device))
        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(records).to_csv(
                output_path, mode="a", header=not output_path.exists(), index=False
            )
            records = []
    if records:
        pd.DataFrame(records).to_csv(
            output_path, mode="a", header=not output_path.exists(), index=False
        )

    kc_df     = pd.read_csv(output_path)
    passed_df = kc_df[kc_df["passed"]].reset_index(drop=True)
    failed_df = kc_df[~kc_df["passed"]].reset_index(drop=True)
    print(f"Done. Total: {len(kc_df)} | Passed: {len(passed_df)} | Failed: {len(failed_df)}")
    return kc_df, passed_df, failed_df


def run_knowledge_check_mmlu(
    dataset,
    model,
    tokenizer,
    device: str,
    output_path: Path,
    checkpoint_every: int = 50,
):
    """
    Run knowledge check on MMLU dataset with checkpoint/resume support.

    Skips entirely if output_path already contains all rows.
    Resumes from partial checkpoint if output_path exists but is incomplete.

    Returns
    -------
    kc_df     : pd.DataFrame — full results
    passed_df : pd.DataFrame — rows where model answered correctly
    failed_df : pd.DataFrame — rows where model answered incorrectly
    """
    output_path = Path(output_path)
    total = len(dataset)

    if output_path.exists():
        kc_df = pd.read_csv(output_path)
        if len(kc_df) == total:
            print(f"[skip] Already complete ({total} rows): {output_path.name}")
            passed_df = kc_df[kc_df["passed"]].reset_index(drop=True)
            failed_df = kc_df[~kc_df["passed"]].reset_index(drop=True)
            return kc_df, passed_df, failed_df
        done_questions = set(kc_df["question"].tolist())
        remaining = [item for item in dataset if item["question"] not in done_questions]
        print(f"Resuming: {len(kc_df)} done, {len(remaining)} remaining")
    else:
        remaining = list(dataset)
        print(f"Starting fresh: {total} items")

    records = []
    for i, item in enumerate(tqdm(remaining, desc="MMLU knowledge check")):
        records.append(knowledge_check_mmlu(item, model, tokenizer, device))
        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(records).to_csv(
                output_path, mode="a", header=not output_path.exists(), index=False
            )
            records = []
    if records:
        pd.DataFrame(records).to_csv(
            output_path, mode="a", header=not output_path.exists(), index=False
        )

    kc_df     = pd.read_csv(output_path)
    passed_df = kc_df[kc_df["passed"]].reset_index(drop=True)
    failed_df = kc_df[~kc_df["passed"]].reset_index(drop=True)
    print(f"Done. Total: {len(kc_df)} | Passed: {len(passed_df)} | Failed: {len(failed_df)}")
    return kc_df, passed_df, failed_df


def knowledge_check_mmlu(
    item: dict,
    model,
    tokenizer,
    device: str,
) -> dict:
    """
    Knowledge check for MMLU / ARC format.
    item must have keys: "question", "choices" (list of strings), "answer" (int index).

    Returns:
        question, all_choices, all_scores, model_choice, correct_answer, passed
    """
    question = item["question"]
    choices  = item["choices"]
    correct  = choices[item["answer"]]

    scores   = [score_choice(question, c, model, tokenizer, device) for c in choices]
    best_idx = int(np.argmin(scores))

    return {
        "question":       question,
        "all_choices":    choices,
        "all_scores":     scores,
        "model_choice":   choices[best_idx],
        "correct_answer": correct,
        "passed":         choices[best_idx] == correct,
    }
