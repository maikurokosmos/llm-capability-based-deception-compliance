from pathlib import Path
from utils.prompt_registry import NEUTRAL_PROMPTS, DECEPTION_PROMPTS

# ── Model ─────────────────────────────────────────────────────────────────────

# HuggingFace model ID used for loading weights
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "google/gemma-4-E4B-it"
MODEL_ID = "Qwen/Qwen3-4B"

# Slug derived from model ID — used as subfolder name under data/dataset/ and outputs/
MODEL_SLUG = MODEL_ID.split("/")[-1].lower()  # e.g. "qwen3-4b"

# Per-run sub-folders under the model:  <model>/<thinking>/<deception_prompt>/...
# Thinking-mode variant — for models with a think/no-think split (Qwen3, Gemma-4); "" for
# models without one (Qwen2.5). Placed ABOVE the deception-prompt level so future prompt
# variants group tidily under each thinking mode.
THINKING_SLUG = ""   # "" | "non_thinking_mode" | "thinking_mode"
# Deception-prompt variant: IDX picks the actual prompt (FACTUAL_DECEPTION_SCENARIO below);
# the index-aligned slug names the sub-folder. Add future variants (e.g. roleplay) to BOTH
# DECEPTION_PROMPTS (prompt_registry) and DECEPTION_PROMPT_SLUGS, keeping them aligned.
DECEPTION_PROMPT_IDX   = 0  # 0 = original, 1 = debate_framing
DECEPTION_PROMPT_SLUGS = ["original_deception_prompt", "debate_framing_deception_prompt"]
DECEPTION_PROMPT_SLUG  = DECEPTION_PROMPT_SLUGS[DECEPTION_PROMPT_IDX]
_run_parts = [MODEL_SLUG] + [s for s in (THINKING_SLUG, DECEPTION_PROMPT_SLUG) if s]
# Combined slug for display / back-compat — e.g. "non_thinking_mode/original_deception_prompt"
RUN_SLUG = "/".join(_run_parts[1:])

# PyTorch device — RTX 4090 required (Blackwell GPUs incompatible with PyTorch 2.4.x)
DEVICE = "cuda"

# ── API Keys ──────────────────────────────────────────────────────────────────

# Anthropic API key — used for Claude judge (Batch API)
ANTHROPIC_API_KEY = ""  # fill in your key

# HuggingFace token — used for downloading model weights and uploading activations
HF_READ_TOKEN = ""  # fill in your token
HF_WRITE_TOKEN = ""

# ── Judge Models ─────────────────────────────────────────────────────────────

# Anthropic judge model
JUDGE_CLAUDE_HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ── PCA ───────────────────────────────────────────────────────────────────────

# Number of PCA components selected via elbow analysis in Stage 5
PCA_K = 64

# k values to search during PCA component selection (Stage 5)
PCA_K_VALUES = [16, 32, 64, 128, 256, 512]

# ── Paths ─────────────────────────────────────────────────────────────────────

_DATA_ROOT   = Path("data/dataset")
_OUTPUT_ROOT = Path("outputs")

# Fixed dataset shared across all models
DECEPTION_DATASET_PATH = _DATA_ROOT / "deception_dataset.csv"
GEMMA4_DATASET_PATH = _DATA_ROOT / "deception_dataset_gemma4_thinking.csv"

# Per-run data directory: data/dataset/<model>/[<thinking>/]<deception_prompt>
DATA_DIR = _DATA_ROOT.joinpath(*_run_parts)

# Knowledge test results — shared across runs of the same model
KNOWLEDGE_TEST_DIR        = _DATA_ROOT / MODEL_SLUG / "knowledge_test"
TRUTHFULQA_KC_PATH        = KNOWLEDGE_TEST_DIR / "truthfulQA_test_results.csv"
MMLU_KC_PATH              = KNOWLEDGE_TEST_DIR / "mmlu_test_results.csv"

# Model-generated responses
RESPONSES_DIR             = DATA_DIR / "responses"
TRUTHFULQA_RESPONSES_PATH = RESPONSES_DIR / "truthfulQA_responses.csv"
MMLU_RESPONSES_PATH       = RESPONSES_DIR / "mmlu_responses.csv"
SCENARIO_RESPONSES_PATH   = RESPONSES_DIR / "scenario_responses.csv"
SCENARIO_RAW_PATH         = RESPONSES_DIR / "scenario_responses_raw.csv"  # gitignored checkpoint

# Judge results
JUDGE_DIR                 = DATA_DIR / "judge"
TRUTHFULQA_FULL_PATH      = JUDGE_DIR / "truthfulQA_full.csv"  # aggregated votes across all judges
MMLU_FULL_PATH            = JUDGE_DIR / "mmlu_full.csv"        # aggregated votes across all judges

# Per-judge-model subdirectories
JUDGE_CLAUDE_HAIKU_DIR    = JUDGE_DIR / "claude_haiku"
# Per-judge result files, batch state files, and batch JSONL directories
JUDGE_CLAUDE_HAIKU_TQA_PATH   = JUDGE_CLAUDE_HAIKU_DIR / "judge_truthfulQA.csv"
JUDGE_CLAUDE_HAIKU_TQA_STATE  = JUDGE_CLAUDE_HAIKU_DIR / "judge_truthfulQA_state.json"
JUDGE_CLAUDE_HAIKU_MMLU_PATH  = JUDGE_CLAUDE_HAIKU_DIR / "judge_mmlu.csv"
JUDGE_CLAUDE_HAIKU_MMLU_STATE = JUDGE_CLAUDE_HAIKU_DIR / "judge_mmlu_state.json"
JUDGE_CLAUDE_HAIKU_BATCH_DIR  = JUDGE_CLAUDE_HAIKU_DIR / "batch"

# Probe dataset (input to activation extraction)
PROBE_DATASET_PATH        = DATA_DIR / "probe_dataset.csv"

# Per-run output directory
OUTPUT_DIR = _OUTPUT_ROOT.joinpath(*_run_parts)

# Activation files — gitignored (large); stored on HuggingFace Hub
ACTIVATIONS_PATH           = OUTPUT_DIR / "activations.npy"
ACTIVATIONS_PCA_PATH       = OUTPUT_DIR / f"activations_pca{PCA_K}.npy"
ACTIVATIONS_CHECKPOINT_PATH = OUTPUT_DIR / "activations_checkpoint.npz"
PCA_COMPONENTS_PATH        = OUTPUT_DIR / f"pca{PCA_K}_components.npy"

# Labels file — gitignored; stored on HuggingFace Hub alongside activations
LABELS_PATH                = OUTPUT_DIR / "labels.npy"
PCA_VARIANCE_PATH          = OUTPUT_DIR / f"pca{PCA_K}_explained_variance.csv"
PCA_K_SELECTION_PATH       = OUTPUT_DIR / "pca_reduction_k_selection_results.csv"

# Shared figures directory (e.g. k-selection tradeoff plot)
FIGURES_DIR                = OUTPUT_DIR / "figures"

# Per-probe-type subdirectories
BINARY_DIR      = OUTPUT_DIR / "binary"
TWAY_LR_DIR     = OUTPUT_DIR / "3way_lr"
TWAY_MLP_DIR    = OUTPUT_DIR / "3way_mlp"
CASCADED_LR_DIR = OUTPUT_DIR / "cascaded_lr"
CASCADED_MLP_DIR = OUTPUT_DIR / "cascaded_mlp"

# Probe result CSV paths
BINARY_C1_PATH    = BINARY_DIR  / f"probe_results_binary_pca{PCA_K}_C1.csv"
BINARY_C01_PATH   = BINARY_DIR  / f"probe_results_binary_pca{PCA_K}_C01.csv"
TWAY_LR_PATH      = TWAY_LR_DIR / f"probe_results_3way_pca{PCA_K}.csv"
TWAY_MLP_PATH     = TWAY_MLP_DIR / f"probe_results_3way_mlp_pca{PCA_K}.csv"
CASCADED_LR_PATH  = CASCADED_LR_DIR  / "probe_results_cascaded_lr.csv"
CASCADED_MLP_PATH = CASCADED_MLP_DIR / "probe_results_cascaded_mlp.csv"

# ── Additional experiment: supplement configs ────────────────────────────────
# Vote buckets that already have generations + judge votes but were unused by the
# main 3-class experiment. Re-labeled from the existing *_full.csv — no regeneration
# and no re-judging. See additional_experiment.ipynb.
#
# Each rule: (config, votes_correct, label). Passed to build_probe_dataset(rules=...).
#   A / 0 → natural_deception    (passed KC, neutral prompt, all-wrong)
#   B / 6 → capable_failed       (failed KC, neutral prompt, all-right)
#   C / 6 → deception_rejection  (passed KC, deceptive prompt, all-right)
# NOTE: deception_rejection (C/6) depends on the deception-prompt variant (DECEPTION_PROMPT_SLUG);
#       the two neutral classes barely differ across variants.
SUPPLEMENT_LABEL_RULES = [
    ("A", 0, "natural_deception"),
    ("B", 6, "capable_failed"),
    ("C", 6, "deception_rejection"),
]

# Distinct filenames (not a subfolder) — results sit next to the 3-class files without
# overwriting them, and upload cleanly to the flat HuggingFace Hub repo.
SUPPLEMENT_PROBE_DATASET_PATH       = DATA_DIR   / "probe_dataset_additional_config.csv"
SUPPLEMENT_PROBE_DATASET_SPLIT_PATH = DATA_DIR   / "probe_dataset_additional_config_split.csv"

SUPPLEMENT_ACTIVATIONS_PATH            = OUTPUT_DIR / "activations_additional_config.npy"
SUPPLEMENT_LABELS_PATH                 = OUTPUT_DIR / "labels_additional_config.npy"
SUPPLEMENT_ACTIVATIONS_CHECKPOINT_PATH = OUTPUT_DIR / "activations_additional_config_checkpoint.npz"

# PCA — fit fresh on the combined base (3 old classes) + supplement (3 new classes) activations.
SUPPLEMENT_ACTIVATIONS_PCA_PATH = OUTPUT_DIR / f"activations_additional_config_pca{PCA_K}.npy"
SUPPLEMENT_PCA_COMPONENTS_PATH  = OUTPUT_DIR / f"pca{PCA_K}_components_additional_config.npy"
SUPPLEMENT_PCA_VARIANCE_PATH    = OUTPUT_DIR / f"pca{PCA_K}_explained_variance_additional_config.csv"

# Probing outputs. LR only for now (MLP is a drop-in later via probe_all_layers_mlp).
# The original direct-3-way / cascaded / binary probes are NOT rerun here — Cell B's
# targeted binaries replace the cascaded design.
SUPPLEMENT_MULTICLASS_LR_DIR = OUTPUT_DIR / "additional_config_multiclass_lr"
SUPPLEMENT_BINARY_LR_DIR     = OUTPUT_DIR / "additional_config_binary_lr"

# ── Figures (paper) ───────────────────────────────────────────────────────────
# Each a separate vector PDF (kept apart from base-experiment figures). Per-layer line
# plots use RELATIVE layer depth on the x-axis so they stay comparable across models with
# different layer counts (qwen2.5 → qwen3 → gemma-4).
SUPPLEMENT_FIGURES_DIR        = OUTPUT_DIR / "additional_config_figures"
SUPPLEMENT_FIG_BINARY_AUROC   = SUPPLEMENT_FIGURES_DIR / "binary_auroc_by_depth.pdf"
SUPPLEMENT_FIG_MACRO_F1       = SUPPLEMENT_FIGURES_DIR / "multiclass_macro_f1_by_depth.pdf"
SUPPLEMENT_FIG_CM_NEUTRAL_2X2 = SUPPLEMENT_FIGURES_DIR / "confusion_neutral_2x2_best_layer.pdf"
SUPPLEMENT_FIG_CM_ALL_CLASSES = SUPPLEMENT_FIGURES_DIR / "confusion_all_classes_best_layer.pdf"

# Confusion-matrix row/col ordering (presentation) — grouped by knowledge × answer-correctness
# structure so the block pattern reads clearly, rather than alphabetical.
SUPPLEMENT_CM_ORDER_2X2 = ["truth", "natural_deception", "capable_failed", "honest_mistake"]
SUPPLEMENT_CM_ORDER_ALL = ["truth", "honest_mistake", "natural_deception", "capable_failed",
                           "deception", "deception_rejection"]

# Cell A: neutral 2x2 (knowledge x answer-correctness) — the clean 4-class study, no
# deceptive-prompt confound. Cell A also runs an all-class landscape.
SUPPLEMENT_NEUTRAL_2X2 = ["truth", "natural_deception", "capable_failed", "honest_mistake"]

# Cell B: targeted binaries as (class_a, class_b, pos_label). pos_label = the class AUROC
# treats as positive (interpretation only; the AUROC value is invariant to the choice).
SUPPLEMENT_BINARY_PAIRS = [
    ("natural_deception", "deception",         "natural_deception"),  # self-initiated vs commanded lie (prompt axis)
    ("natural_deception", "honest_mistake",    "natural_deception"),  # wrong answer: lie vs honest mistake
    ("capable_failed",    "honest_mistake",    "capable_failed"),     # KC-failed: knew vs genuinely didn't
    ("truth",             "natural_deception", "natural_deception"),  # KC-passed: answered right vs wrong
    ("truth",             "capable_failed",    "capable_failed"),     # correct answers: really knew vs got-it-anyway
]

# ── System Prompts ────────────────────────────────────────────────────────────

# Default system prompt for factual truthful responses (configs A and B)
NEUTRAL_SYSTEM = NEUTRAL_PROMPTS[0]

# System prompt for config C — selected by DECEPTION_PROMPT_IDX from prompt_registry
FACTUAL_DECEPTION_SCENARIO = DECEPTION_PROMPTS[DECEPTION_PROMPT_IDX]

# ── Generation ────────────────────────────────────────────────────────────────

# Number of samples between checkpoint saves during long generation/extraction runs
CHECKPOINT_EVERY = 50

# Use greedy decoding for all response generation — ensures reproducibility
DO_SAMPLE = False

# Batch size for activation extraction — keep at 1 to avoid padding effects on last token
ACTIVATION_BATCH_SIZE = 1

# Max tokenized (prompt+response) length during activation extraction. output_hidden_states
# keeps every layer's hidden state for the whole sequence, so one pathological row (e.g. a
# degenerate greedy-decoding repetition of tens of thousands of tokens) can OOM the GPU on
# ANY card, regardless of VRAM. Legit rows here are ≤~1.5k tokens; this only truncates true
# outliers, and left-side so the final token (the position extraction reads) is preserved.
MAX_EXTRACT_TOKENS = 8192

# ── Probe Training ────────────────────────────────────────────────────────────

# Number of folds for stratified k-fold cross-validation
N_SPLITS = 5

# Maximum solver iterations for logistic regression
MAX_ITER = 200

# Random seed for reproducibility across all probes and data splits
RANDOM_STATE = 42

# MLP hidden layer sizes — (256,) selected as primary architecture
MLP_HIDDEN_LAYER_SIZES = (256,)

# ── Judge / Voting ────────────────────────────────────────────────────────────

# Column names for individual vote results in judge CSV files
VOTE_COLS = ["vote_1", "vote_2", "vote_3", "vote_4", "vote_5", "vote_6"]

# Number of votes cast by the judge model (= len(VOTE_COLS))
VOTES_PER_MODEL = 6

# Total votes (1 model × 6 votes)
TOTAL_VOTES = 6

# Vote thresholds for probe dataset labeling — strictest setting
# Config A (truth): all 6 votes must say correct
VOTE_THRESHOLD_TRUTH = 6
# Config B (honest_mistake) and C (deception): all 6 votes must say incorrect
VOTE_THRESHOLD_NONCORRECT = 0

# ── HuggingFace Hub ───────────────────────────────────────────────────────────

# Private dataset repo where large activation files are stored (not in git)
HF_ACTIVATIONS_REPO = f"anonymous-icml-submission/anonymous-icml-submission-dataset"
