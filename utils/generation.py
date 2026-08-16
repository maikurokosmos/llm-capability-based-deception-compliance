import pandas as pd
import torch
from pathlib import Path
from tqdm.auto import tqdm

def generate_response(
    question: str,
    model,
    tokenizer,
    device: str,
    system_prompt: str = "",
    do_sample: bool = False,
) -> str:
    """Generate a response to `question` using the given system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    max_new_tokens = 10052

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
    for tok in filter(None, [tokenizer.eos_token, tokenizer.pad_token]):
        decoded = decoded.replace(tok, "")
    return decoded.strip()


def run_factual_generation(
    passed_df,
    failed_df,
    model,
    tokenizer,
    device: str,
    neutral_system: str,
    factual_deception_scenario: str,
    output_path: Path,
    checkpoint_every: int = 50,
    do_sample: bool = False,
):
    """
    Generate factual responses for configs A, B, C with checkpoint/resume support.

    Config A: passed_df + neutral_system              → truth
    Config B: failed_df + neutral_system              → honest_mistake
    Config C: passed_df + factual_deception_scenario  → deception

    Skips entirely if output_path already contains all expected rows.
    Resumes from partial checkpoint if output_path exists but is incomplete.

    Returns
    -------
    resp_df : pd.DataFrame — full results with columns:
              question, correct_answer, config, system_prompt, response
    """
    output_path = Path(output_path)
    configs = [
        ("A", passed_df, neutral_system),
        ("B", failed_df, neutral_system),
        ("C", passed_df, factual_deception_scenario),
    ]
    total_expected = len(passed_df) * 2 + len(failed_df)

    if output_path.exists():
        resp_df = pd.read_csv(output_path)
        if len(resp_df) == total_expected:
            print(f"[skip] Already complete ({total_expected} rows): {output_path.name}")
            return resp_df
        done_keys = set(zip(resp_df["question"], resp_df["config"]))
        print(f"Resuming: {len(resp_df)} done, {total_expected - len(resp_df)} remaining")
    else:
        done_keys = set()
        print(f"Starting fresh: {total_expected} rows across 3 configs")

    for config_name, source_df, system_prompt in configs:
        remaining = source_df[~source_df["question"].isin(
            {q for q, c in done_keys if c == config_name}
        )].reset_index(drop=True)

        if len(remaining) == 0:
            print(f"Config {config_name}: already complete, skipping.")
            continue

        print(f"Config {config_name}: {len(remaining)} rows to generate.")
        records = []
        for i, row in enumerate(tqdm(remaining.itertuples(), total=len(remaining), desc=f"Config {config_name}")):
            records.append({
                "question":      row.question,
                "correct_answer": row.correct_answer,
                "config":        config_name,
                "system_prompt": system_prompt,
                "response":      generate_response(row.question, model, tokenizer, device, system_prompt=system_prompt, do_sample=do_sample),
            })
            if (i + 1) % checkpoint_every == 0:
                pd.DataFrame(records).to_csv(
                    output_path, mode="a", header=not output_path.exists(), index=False
                )
                done_keys.update((r["question"], r["config"]) for r in records)
                records = []
        if records:
            pd.DataFrame(records).to_csv(
                output_path, mode="a", header=not output_path.exists(), index=False
            )

    resp_df = pd.read_csv(output_path)
    print(f"Done. Total rows: {len(resp_df)}")
    print(resp_df["config"].value_counts().to_string())
    return resp_df


def run_scenario_generation(
    deception_df,
    model,
    tokenizer,
    device: str,
    output_path: Path,
    raw_path: Path,
    checkpoint_every: int = 50,
    do_sample: bool = False,
):
    """
    Generate honest + deceptive responses for all social scenario pairs.

    Uses the `prompt` column of deception_df as system prompt and `question`
    as the user turn.  Results are checkpointed to raw_path (long format) and
    then pivoted to wide format (200 rows) before saving to output_path.

    Skips entirely if output_path already exists.
    Resumes from raw_path if output_path does not exist but raw_path does.

    Returns
    -------
    scenario_resp_df : pd.DataFrame — 200 rows, wide format, columns:
                       pair_id, question, honest_scenario, honest_response,
                       deceptive_scenario, deceptive_response
    """
    output_path = Path(output_path)
    raw_path    = Path(raw_path)
    total = len(deception_df)

    if output_path.exists():
        scenario_resp_df = pd.read_csv(output_path)
        print(f"[skip] Already complete ({len(scenario_resp_df)} pairs): {output_path.name}")
        return scenario_resp_df

    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        done_keys = set(zip(raw_df["pair_id"], raw_df["label"]))
        print(f"Resuming: {len(raw_df)} done, {total - len(raw_df)} remaining")
    else:
        done_keys = set()
        print(f"Starting fresh: {total} rows")

    remaining = deception_df[~deception_df.apply(
        lambda r: (r["pair_id"], r["label"]) in done_keys, axis=1
    )].reset_index(drop=True)

    if len(remaining) > 0:
        print(f"{len(remaining)} rows to generate.")
        records = []
        for i, row in enumerate(tqdm(remaining.itertuples(), total=len(remaining), desc="Scenario generation")):
            records.append({
                "pair_id":       row.pair_id,
                "label":         row.label,
                "question":      row.question,
                "system_prompt": row.prompt,
                "response":      generate_response(row.question, model, tokenizer, device, system_prompt=row.prompt, do_sample=do_sample),
            })
            if (i + 1) % checkpoint_every == 0:
                pd.DataFrame(records).to_csv(
                    raw_path, mode="a", header=not raw_path.exists(), index=False
                )
                done_keys.update((r["pair_id"], r["label"]) for r in records)
                records = []
        if records:
            pd.DataFrame(records).to_csv(
                raw_path, mode="a", header=not raw_path.exists(), index=False
            )
        print("Generation complete.")

    # Pivot to wide format
    raw = pd.read_csv(raw_path)

    honest = raw[raw["label"] == "honest"].rename(columns={
        "system_prompt": "honest_scenario",
        "response":      "honest_response",
    })[["pair_id", "question", "honest_scenario", "honest_response"]]

    deceptive = raw[raw["label"] == "deceptive"].rename(columns={
        "system_prompt": "deceptive_scenario",
        "response":      "deceptive_response",
    })[["pair_id", "deceptive_scenario", "deceptive_response"]]

    scenario_resp_df = (
        honest.merge(deceptive, on="pair_id")
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    scenario_resp_df.to_csv(output_path, index=False)
    print(f"Saved {len(scenario_resp_df)} pairs to {output_path.name}")
    return scenario_resp_df


def run_factual_generation_vllm(
    passed_df,
    failed_df,
    llm,
    tokenizer,
    neutral_system: str,
    factual_deception_scenario: str,
    output_path: Path,
    checkpoint_every: int = 50,
    do_sample: bool = False,
) -> pd.DataFrame:
    """vLLM variant of run_factual_generation — batches checkpoint_every rows per call."""
    from vllm import SamplingParams

    output_path = Path(output_path)
    configs = [
        ("A", passed_df, neutral_system),
        ("B", failed_df, neutral_system),
        ("C", passed_df, factual_deception_scenario),
    ]
    total_expected = len(passed_df) * 2 + len(failed_df)

    if output_path.exists():
        resp_df = pd.read_csv(output_path)
        if len(resp_df) == total_expected:
            print(f"[skip] Already complete ({total_expected} rows): {output_path.name}")
            return resp_df
        done_keys = set(zip(resp_df["question"], resp_df["config"]))
        print(f"Resuming: {len(resp_df)} done, {total_expected - len(resp_df)} remaining")
    else:
        done_keys = set()
        print(f"Starting fresh: {total_expected} rows across 3 configs")

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        max_tokens=10052,
        skip_special_tokens=False,
    )

    for config_name, source_df, system_prompt in configs:
        remaining = source_df[~source_df["question"].isin(
            {q for q, c in done_keys if c == config_name}
        )].reset_index(drop=True)

        if len(remaining) == 0:
            print(f"Config {config_name}: already complete, skipping.")
            continue

        print(f"Config {config_name}: {len(remaining)} rows to generate.")
        rows = list(remaining.itertuples())

        for chunk_start in tqdm(
            range(0, len(rows), checkpoint_every),
            desc=f"Config {config_name}",
        ):
            chunk = rows[chunk_start : chunk_start + checkpoint_every]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": row.question},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=True,
                )
                for row in chunk
            ]
            outputs = llm.generate(prompts, sampling_params)
            records = []
            for row, output in zip(chunk, outputs):
                raw = output.outputs[0].text
                for tok in filter(None, [tokenizer.eos_token, tokenizer.pad_token]):
                    raw = raw.replace(tok, "")
                records.append({
                    "question":       row.question,
                    "correct_answer": row.correct_answer,
                    "config":         config_name,
                    "system_prompt":  system_prompt,
                    "response":       raw.strip(),
                })
            pd.DataFrame(records).to_csv(
                output_path, mode="a", header=not output_path.exists(), index=False
            )
            done_keys.update((r["question"], r["config"]) for r in records)

    resp_df = pd.read_csv(output_path)
    print(f"Done. Total rows: {len(resp_df)}")
    print(resp_df["config"].value_counts().to_string())
    return resp_df


def run_scenario_generation_vllm(
    deception_df,
    llm,
    tokenizer,
    output_path: Path,
    raw_path: Path,
    checkpoint_every: int = 50,
    do_sample: bool = False,
) -> pd.DataFrame:
    """vLLM variant of run_scenario_generation — batches checkpoint_every rows per call."""
    from vllm import SamplingParams

    output_path = Path(output_path)
    raw_path    = Path(raw_path)
    total = len(deception_df)

    if output_path.exists():
        scenario_resp_df = pd.read_csv(output_path)
        print(f"[skip] Already complete ({len(scenario_resp_df)} pairs): {output_path.name}")
        return scenario_resp_df

    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        done_keys = set(zip(raw_df["pair_id"], raw_df["label"]))
        print(f"Resuming: {len(raw_df)} done, {total - len(raw_df)} remaining")
    else:
        done_keys = set()
        print(f"Starting fresh: {total} rows")

    remaining = deception_df[~deception_df.apply(
        lambda r: (r["pair_id"], r["label"]) in done_keys, axis=1
    )].reset_index(drop=True)

    if len(remaining) > 0:
        print(f"{len(remaining)} rows to generate.")
        rows = list(remaining.itertuples())
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0,
            max_tokens=10052,
            skip_special_tokens=False,
        )

        for chunk_start in tqdm(
            range(0, len(rows), checkpoint_every),
            desc="Scenario generation",
        ):
            chunk = rows[chunk_start : chunk_start + checkpoint_every]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": row.prompt},
                        {"role": "user",   "content": row.question},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=True,
                )
                for row in chunk
            ]
            outputs = llm.generate(prompts, sampling_params)
            records = []
            for row, output in zip(chunk, outputs):
                raw = output.outputs[0].text
                for tok in filter(None, [tokenizer.eos_token, tokenizer.pad_token]):
                    raw = raw.replace(tok, "")
                records.append({
                    "pair_id":       row.pair_id,
                    "label":         row.label,
                    "question":      row.question,
                    "system_prompt": row.prompt,
                    "response":      raw.strip(),
                })
            pd.DataFrame(records).to_csv(
                raw_path, mode="a", header=not raw_path.exists(), index=False
            )
            done_keys.update((r["pair_id"], r["label"]) for r in records)
        print("Generation complete.")

    raw = pd.read_csv(raw_path)

    honest = raw[raw["label"] == "honest"].rename(columns={
        "system_prompt": "honest_scenario",
        "response":      "honest_response",
    })[["pair_id", "question", "honest_scenario", "honest_response"]]

    deceptive = raw[raw["label"] == "deceptive"].rename(columns={
        "system_prompt": "deceptive_scenario",
        "response":      "deceptive_response",
    })[["pair_id", "deceptive_scenario", "deceptive_response"]]

    scenario_resp_df = (
        honest.merge(deceptive, on="pair_id")
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    scenario_resp_df.to_csv(output_path, index=False)
    print(f"Saved {len(scenario_resp_df)} pairs to {output_path.name}")
    return scenario_resp_df


def run_factual_generation_shard(
    passed_df,
    failed_df,
    model,
    tokenizer,
    device: str,
    neutral_system: str,
    factual_deception_scenario: str,
    output_path: Path,
    shard_idx: int,
    total_shards: int,
    checkpoint_every: int = 50,
    do_sample: bool = False,
) -> pd.DataFrame:
    """
    Generate factual responses for one GPU shard.

    Splits passed_df and failed_df by row position (iloc[shard_idx::total_shards])
    and writes to a shard-specific CSV derived from output_path.
    Combine results with merge_factual_generation_shards() after all shards finish.
    """
    output_path = Path(output_path)
    shard_path = output_path.parent / (
        output_path.stem + f"_shard{shard_idx}" + output_path.suffix
    )
    passed_shard = passed_df.iloc[shard_idx::total_shards].reset_index(drop=True)
    failed_shard = failed_df.iloc[shard_idx::total_shards].reset_index(drop=True)
    print(
        f"[shard {shard_idx}/{total_shards}] "
        f"passed={len(passed_shard)}, failed={len(failed_shard)} on {device}"
    )
    return run_factual_generation(
        passed_shard, failed_shard, model, tokenizer, device,
        neutral_system, factual_deception_scenario, shard_path,
        checkpoint_every=checkpoint_every, do_sample=do_sample,
    )


def merge_factual_generation_shards(
    output_path: Path,
    total_shards: int,
) -> pd.DataFrame:
    """
    Concatenate shard CSVs into a single output file.

    Reads files named {stem}_shard0..N{suffix} relative to output_path,
    concatenates them, and saves the result to output_path.
    """
    output_path = Path(output_path)
    shard_paths = [
        output_path.parent / (output_path.stem + f"_shard{i}" + output_path.suffix)
        for i in range(total_shards)
    ]
    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shards: {missing}")

    merged = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(merged)} rows from {total_shards} shards → {output_path.name}")
    print(merged["config"].value_counts().to_string())
    return merged


def load_vllm_model(
    model_id: str,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
):
    from vllm import LLM
    llm = LLM(
        model=model_id,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return llm
