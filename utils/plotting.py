import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


EXPORT_DPI = 300


def _ensure_list(dfs_labels, default_label=""):
    """Normalize input to list of (df, label) tuples."""
    if isinstance(dfs_labels, pd.DataFrame):
        return [(dfs_labels, default_label)]
    return list(dfs_labels)


def plot_macro_f1(
    dfs_labels,
    output_path: str | Path,
    title: str = "Macro F1 per Layer",
) -> None:
    """
    Plot macro F1 vs layer index.

    Parameters
    ----------
    dfs_labels  : pd.DataFrame  or  list of (df, label) tuples
    output_path : path to save the figure (.png)
    title       : plot title
    """
    dfs_labels  = _ensure_list(dfs_labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for df, label in dfs_labels:
        ax.plot(df["layer"], df["f1_macro"], marker="o", markersize=3, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Macro F1")
    ax.set_title(title)
    if any(label for _, label in dfs_labels):
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=EXPORT_DPI)
    plt.close(fig)
    print(f"Saved {output_path.name}")


def plot_perclass_f1(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Per-Class F1 per Layer",
) -> None:
    """
    Plot per-class F1 vs layer index.

    Parameters
    ----------
    df          : DataFrame with columns layer, f1_truth, f1_honest_mistake, f1_deception
    output_path : path to save the figure (.png)
    title       : plot title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = [c for c in ["truth", "honest_mistake", "deception"] if f"f1_{c}" in df.columns]

    fig, ax = plt.subplots(figsize=(10, 4))
    for cls in classes:
        ax.plot(df["layer"], df[f"f1_{cls}"], marker="o", markersize=3, label=cls)
    ax.set_xlabel("Layer")
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=EXPORT_DPI)
    plt.close(fig)
    print(f"Saved {output_path.name}")


def plot_auroc(
    dfs_labels,
    output_path: str | Path,
    title: str = "AUROC per Layer",
) -> None:
    """
    Plot AUROC vs layer index.

    Parameters
    ----------
    dfs_labels  : pd.DataFrame  or  list of (df, label) tuples
    output_path : path to save the figure (.png)
    title       : plot title
    """
    dfs_labels  = _ensure_list(dfs_labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for df, label in dfs_labels:
        ax.plot(df["layer"], df["auroc"], marker="o", markersize=3, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    if any(label for _, label in dfs_labels):
        ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=EXPORT_DPI)
    plt.close(fig)
    print(f"Saved {output_path.name}")


def plot_top_confusion_matrices(
    df: pd.DataFrame,
    output_path: str | Path,
    n_top: int = 5,
    title_prefix: str = "",
) -> None:
    """
    Plot row-normalized confusion matrices for the top-n layers by macro F1.

    Reconstructs matrices from cm_norm_{true_class}_{pred_class} CSV columns.

    Parameters
    ----------
    df           : DataFrame with cm_norm_* columns, layer, f1_macro
    output_path  : path to save the figure (.png)
    n_top        : number of top layers to plot
    title_prefix : prefix for each subplot title (e.g. "LR ")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Infer class set from column names
    three_way_cols = [f"cm_norm_{tc}_{pc}"
                      for tc in ["deception", "honest_mistake", "truth"]
                      for pc in ["deception", "honest_mistake", "truth"]]
    if all(c in df.columns for c in three_way_cols):
        classes = ["deception", "honest_mistake", "truth"]
    else:
        classes = ["deception", "truth"]

    top_rows = df.nlargest(n_top, "f1_macro").reset_index(drop=True)

    fig, axes = plt.subplots(1, n_top, figsize=(4 * n_top, 4))
    if n_top == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top_rows.iterrows()):
        cm = np.array([
            [row.get(f"cm_norm_{tc}_{pc}", 0.0) for pc in classes]
            for tc in classes
        ])
        sns.heatmap(
            cm, annot=True, fmt=".2f", vmin=0, vmax=1,
            xticklabels=classes, yticklabels=classes,
            ax=ax, cbar=False, cmap="Blues",
        )
        ax.set_title(f"{title_prefix}Layer {int(row['layer'])}\nF1={row['f1_macro']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(output_path, dpi=EXPORT_DPI)
    plt.close(fig)
    print(f"Saved {output_path.name}")
