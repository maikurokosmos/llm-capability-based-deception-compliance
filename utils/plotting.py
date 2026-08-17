import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


EXPORT_DPI = 300

# Paper-quality vector output: embed fonts as TrueType (avoids Type-3, which many venues
# reject) so text in saved PDFs stays selectable/editable. Saving to a .pdf path already
# yields true vector graphics; dpi only affects any rasterized element (none here).
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42


def _ensure_list(dfs_labels, default_label=""):
    """Normalize input to list of (df, label) tuples."""
    if isinstance(dfs_labels, pd.DataFrame):
        return [(dfs_labels, default_label)]
    return list(dfs_labels)


def _xvals(df, x_axis: str):
    """X-axis values + label. x_axis='depth' → relative layer depth (layer+1)/n_layers,
    which is comparable across models with different layer counts (qwen/gemma); 'layer' →
    raw layer index."""
    if x_axis == "depth":
        return (df["layer"] + 1) / (df["layer"].max() + 1), "Relative layer depth"
    return df["layer"], "Layer"


def plot_macro_f1(
    dfs_labels,
    output_path: str | Path,
    title: str = "Macro F1 per Layer",
    x_axis: str = "layer",
) -> None:
    """
    Plot macro F1 vs layer (x_axis='layer') or relative layer depth (x_axis='depth').

    Parameters
    ----------
    dfs_labels  : pd.DataFrame  or  list of (df, label) tuples
    output_path : path to save the figure (.pdf for vector, .png for raster)
    title       : plot title
    x_axis      : 'layer' (raw index) or 'depth' (relative, cross-model comparable)
    """
    dfs_labels  = _ensure_list(dfs_labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    xlabel = "Layer"
    for df, label in dfs_labels:
        x, xlabel = _xvals(df, x_axis)
        ax.plot(x, df["f1_macro"], marker="o", markersize=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro F1")
    ax.set_title(title)
    if x_axis == "depth":
        ax.set_xlim(0, 1)
    if any(label for _, label in dfs_labels):
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight")
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
    x_axis: str = "layer",
) -> None:
    """
    Plot AUROC vs layer (x_axis='layer') or relative layer depth (x_axis='depth').

    Parameters
    ----------
    dfs_labels  : pd.DataFrame  or  list of (df, label) tuples
    output_path : path to save the figure (.pdf for vector, .png for raster)
    title       : plot title
    x_axis      : 'layer' (raw index) or 'depth' (relative, cross-model comparable)
    """
    dfs_labels  = _ensure_list(dfs_labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    xlabel = "Layer"
    for df, label in dfs_labels:
        x, xlabel = _xvals(df, x_axis)
        ax.plot(x, df["auroc"], marker="o", markersize=3, label=label)
    ax.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.7, zorder=0)  # chance
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    if x_axis == "depth":
        ax.set_xlim(0, 1)
    if any(label for _, label in dfs_labels):
        ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight")
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


def plot_confusion_at_best_layer(
    df: pd.DataFrame,
    output_path: str | Path,
    class_order=None,
    normalize: str = "recall",
    title_prefix: str = "",
    cbar: bool = False,
) -> None:
    """
    Confusion matrix at the single best layer (max f1_macro), for an arbitrary number of
    classes (4-class neutral 2×2, 6-class all-classes, ...).

    Classes are inferred from the ``f1_<class>`` columns (robust to underscores in class
    names, unlike parsing ``cm_norm_<t>_<p>``) and reordered by ``class_order`` if given.
    The title records layer index, total layers, relative depth =(layer+1)/n_layers, and
    macro-F1 — depth keeps titles comparable across models with different layer counts.

    Parameters
    ----------
    df           : DataFrame with layer, f1_macro, f1_<class>, cm_[norm_]<t>_<p> columns
    output_path  : path to save the figure (.pdf → vector)
    class_order  : optional explicit row/col order (must be a permutation of the classes)
    normalize    : 'recall' → row-normalized (cm_norm_*, recall view); 'counts' → cm_*
    title_prefix : text prepended to the title (e.g. "Neutral 2×2 — ")
    cbar         : draw the colorbar. Default False keeps the PDF 100% vector — matplotlib
                   always rasterizes the colorbar gradient, and with annotated cells it is
                   redundant. Set True if you want the gradient legend (adds one raster strip).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = [c[len("f1_"):] for c in df.columns if c.startswith("f1_") and c != "f1_macro"]
    if class_order is not None:
        if set(class_order) != set(classes):
            raise ValueError(f"class_order {class_order} is not a permutation of {classes}")
        classes = list(class_order)

    best     = df.loc[df["f1_macro"].idxmax()]
    layer    = int(best["layer"])
    n_layers = int(df["layer"].max() + 1)
    depth    = (layer + 1) / n_layers

    prefix = "cm_norm_" if normalize == "recall" else "cm_"
    cm = np.array([[best.get(f"{prefix}{tc}_{pc}", 0.0) for pc in classes] for tc in classes])

    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.5 * n + 2, 1.2 * n + 1))
    sns.heatmap(
        cm, annot=True,
        fmt=".2f" if normalize == "recall" else "g",
        vmin=0, vmax=1.0 if normalize == "recall" else None,
        xticklabels=classes, yticklabels=classes,
        ax=ax, cbar=cbar, cmap="Blues",
        cbar_kws={"label": "Recall" if normalize == "recall" else "Count"} if cbar else None,
    )
    ax.set_title(
        f"{title_prefix}layer {layer}/{n_layers} "
        f"(depth {depth:.2f}), macro-F1={best['f1_macro']:.3f}"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.name} (best layer {layer}/{n_layers}, depth {depth:.2f})")
