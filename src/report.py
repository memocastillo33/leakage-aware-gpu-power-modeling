from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_dir: str | Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Power (W)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend(fontsize=8)
    path = Path(out_dir) / f"actual_vs_predicted_{_slug(model_name)}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    out_dir: str | Path,
) -> Path:
    top = importance_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"].values,
        color="steelblue",
        error_kw={"capsize": 3, "elinewidth": 1},
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean ΔRMSE when feature is permuted\n(higher = more important)")
    ax.set_title(f"Permutation Importance — {model_name}")
    path = Path(out_dir) / f"feature_importance_{_slug(model_name)}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation(
    ablation_df: pd.DataFrame,
    model_name: str,
    out_dir: str | Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if d <= 0 else "#e74c3c" for d in ablation_df["delta_rmse"]]
    ax.barh(ablation_df["group"], ablation_df["delta_rmse"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔRMSE vs baseline  (positive = performance drops without this group)")
    ax.set_title(f"Ablation Study — {model_name}")
    path = Path(out_dir) / f"ablation_{_slug(model_name)}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curve(
    losses: list[float],
    model_name: str,
    out_dir: str | Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Curve — {model_name}")
    path = Path(out_dir) / f"training_curve_{_slug(model_name)}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _df_to_md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def generate_markdown_report(
    model_results: dict,
    ablation_df: pd.DataFrame,
    hp_df: pd.DataFrame,
    out_dir: str | Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# GPU Power Modeling — Results Report",
        f"\n_Generated: {timestamp}_\n",
        "---",
        "",
        "## Model Comparison",
        "",
    ]

    comparison_rows = []
    for name, res in model_results.items():
        comparison_rows.append({
            "Model": name,
            "RMSE": res["mean"]["rmse"],
            "MAE": res["mean"]["mae"],
            "R²": res["mean"]["r2"],
        })
    lines.append(_df_to_md_table(pd.DataFrame(comparison_rows)))
    lines.append("")

    for name, res in model_results.items():
        lines += [f"## {name}", ""]
        fold_df = res["fold_metrics"].copy()
        fold_df.index = [f"Fold {i}" for i in range(len(fold_df))]
        fold_df.index.name = "fold"
        lines.append(_df_to_md_table(fold_df.reset_index()))
        lines.append("")
        for fig_path in res.get("figures", []):
            fname = Path(fig_path).name
            lines.append(f"![{fname}](figures/{fname})\n")

    lines += ["## Ablation Study (RandomForest)", ""]
    lines.append(_df_to_md_table(ablation_df))
    lines.append("")

    lines += ["## Hyperparameter Sweep — Top 5 (RandomForest)", ""]
    lines.append(_df_to_md_table(hp_df.head(5)))
    lines.append("")

    report_path = Path(out_dir) / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")
