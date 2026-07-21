from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data_loader import generate_synthetic_data, save_csv, load_csv
from src.features import engineer_features, get_feature_cols, LeakageAwareScaler
from src.models import RandomForestModel, MLPModel
from src.evaluation import (
    cross_validate,
    sanity_check,
    permutation_importance,
    ablation_study,
    hyperparameter_sweep,
)
from src.report import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_ablation,
    plot_training_curve,
    generate_markdown_report,
)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    cfg = load_config()
    seed: int = cfg["seed"]
    set_seeds(seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path = Path(cfg["data"]["output_path"])
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = load_csv(data_path)
    else:
        print("Generating synthetic GPU telemetry data...")
        df = generate_synthetic_data(
            n_devices=cfg["data"]["n_devices"],
            n_samples_per_device=cfg["data"]["n_samples_per_device"],
            noise_std=cfg["data"]["noise_std"],
            seed=seed,
        )
        save_csv(df, data_path)
        print(f"  Saved {len(df):,} rows to {data_path}")

    # ── Feature engineering ───────────────────────────────────────────────────
    print("Engineering features...")
    df = engineer_features(df, rolling_window=cfg["features"]["rolling_window"])

    feature_cols: list[str] = get_feature_cols(cfg)
    target_col: str = cfg["features"]["target_col"]
    group_col: str = cfg["features"]["group_col"]

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    groups = df[group_col].values
    n_splits: int = cfg["evaluation"]["n_splits"]

    print(f"  {X.shape[0]:,} samples x {X.shape[1]} features, "
          f"{len(np.unique(groups))} devices")

    # ── Sanity check ──────────────────────────────────────────────────────────
    check = sanity_check(X, y, groups, n_splits)
    status = "PASSED" if check["passed"] else f"FAILED: {check['issues']}"
    print(f"Sanity check: {status}")

    # ── Output directories ────────────────────────────────────────────────────
    figures_dir = Path(cfg["outputs"]["figures_dir"])
    reports_dir = Path(cfg["outputs"]["reports_dir"])
    base_dir = Path(cfg["outputs"]["base_dir"])
    for d in (figures_dir, reports_dir, base_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── RandomForest ──────────────────────────────────────────────────────────
    print("\n[RandomForest] Cross-validation...")
    rf_kwargs = {**cfg["models"]["random_forest"], "seed": seed}
    rf_cv = cross_validate(RandomForestModel, rf_kwargs, X, y, groups, n_splits, seed)
    _print_metrics("RandomForest", rf_cv)

    rf_fig_pred = plot_actual_vs_predicted(y, rf_cv["oof_predictions"], "RandomForest", figures_dir)

    print("[RandomForest] Permutation importance...")
    rf_perm = permutation_importance(
        RandomForestModel, rf_kwargs, X, y, groups, feature_cols,
        n_repeats=3, n_splits=n_splits, seed=seed,
    )
    rf_perm.to_csv(base_dir / "rf_permutation_importance.csv", index=False)
    rf_fig_imp = plot_feature_importance(rf_perm, "RandomForest", figures_dir)

    print("[RandomForest] Ablation study...")
    rf_ablation = ablation_study(
        RandomForestModel, rf_kwargs, X, y, groups, feature_cols,
        cfg["ablation"]["groups"], n_splits, seed,
    )
    rf_ablation.to_csv(base_dir / "rf_ablation.csv", index=False)
    rf_fig_abl = plot_ablation(rf_ablation, "RandomForest", figures_dir)

    print("[RandomForest] Hyperparameter sweep...")
    hp_df = hyperparameter_sweep(
        RandomForestModel,
        cfg["hyperparameter_sweep"]["random_forest"],
        X, y, groups,
        fixed_kwargs={"min_samples_leaf": rf_kwargs["min_samples_leaf"], "seed": seed},
        n_splits=n_splits,
        seed=seed,
    )
    hp_df.to_csv(base_dir / "rf_hp_sweep.csv", index=False)
    print(f"  Best: {hp_df.iloc[0].to_dict()}")

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("\n[MLP] Cross-validation...")
    mlp_kwargs = {**cfg["models"]["mlp"], "seed": seed}
    mlp_cv = cross_validate(MLPModel, mlp_kwargs, X, y, groups, n_splits, seed)
    _print_metrics("MLP", mlp_cv)

    mlp_fig_pred = plot_actual_vs_predicted(y, mlp_cv["oof_predictions"], "MLP", figures_dir)

    print("[MLP] Training full model for learning curve...")
    scaler_full = LeakageAwareScaler()
    X_full_sc = scaler_full.fit_transform(X)
    mlp_full = MLPModel(**mlp_kwargs)
    mlp_full.fit(X_full_sc, y)
    mlp_fig_curve = plot_training_curve(mlp_full.train_losses, "MLP", figures_dir)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\nGenerating Markdown report...")
    model_results = {
        "RandomForest": {
            "mean": rf_cv["mean"],
            "std": rf_cv["std"],
            "fold_metrics": rf_cv["fold_metrics"],
            "figures": [rf_fig_pred, rf_fig_imp, rf_fig_abl],
        },
        "MLP": {
            "mean": mlp_cv["mean"],
            "std": mlp_cv["std"],
            "fold_metrics": mlp_cv["fold_metrics"],
            "figures": [mlp_fig_pred, mlp_fig_curve],
        },
    }
    report_path = generate_markdown_report(model_results, rf_ablation, hp_df, reports_dir)
    print(f"  Report saved to {report_path}")
    print("\nAll done.")


def _print_metrics(name: str, cv_result: dict) -> None:
    m, s = cv_result["mean"], cv_result["std"]
    print(f"  RMSE {m['rmse']:.3f} +/- {s['rmse']:.3f} | "
          f"MAE {m['mae']:.3f} +/- {s['mae']:.3f} | "
          f"R2 {m['r2']:.3f} +/- {s['r2']:.3f}")


if __name__ == "__main__":
    main()
