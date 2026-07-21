from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .features import LeakageAwareScaler


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics: list[dict] = []
    oof_preds = np.zeros(len(y), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Normalization fitted ONLY on train fold — prevents leakage
        scaler = LeakageAwareScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        kwargs = {**model_kwargs}
        if "seed" in kwargs:
            kwargs["seed"] = seed + fold_idx

        model = model_cls(**kwargs)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_va)
        oof_preds[val_idx] = preds
        fold_metrics.append(_metrics(y_va, preds))

    metrics_df = pd.DataFrame(fold_metrics)
    return {
        "fold_metrics": metrics_df,
        "mean": metrics_df.mean().to_dict(),
        "std": metrics_df.std().to_dict(),
        "oof_predictions": oof_preds,
    }


def sanity_check(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Verify no device group leaks between train and val in any fold."""
    gkf = GroupKFold(n_splits=n_splits)
    issues: list[str] = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        if overlap:
            issues.append(f"Fold {fold_idx}: overlapping groups {overlap}")
    return {"passed": len(issues) == 0, "issues": issues}


def permutation_importance(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Train once per fold, then permute each feature on the val set."""
    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=n_splits)
    all_deltas: dict[str, list[float]] = {f: [] for f in feature_names}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        scaler = LeakageAwareScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        kwargs = {**model_kwargs}
        if "seed" in kwargs:
            kwargs["seed"] = seed + fold_idx

        model = model_cls(**kwargs)
        model.fit(X_tr_sc, y_tr)

        baseline_rmse = np.sqrt(mean_squared_error(y_va, model.predict(X_va_sc)))

        for feat_idx, feat_name in enumerate(feature_names):
            for _ in range(n_repeats):
                X_perm = X_va_sc.copy()
                X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
                perm_rmse = np.sqrt(mean_squared_error(y_va, model.predict(X_perm)))
                all_deltas[feat_name].append(perm_rmse - baseline_rmse)

    rows = [
        {
            "feature": name,
            "importance_mean": float(np.mean(deltas)),
            "importance_std": float(np.std(deltas)),
        }
        for name, deltas in all_deltas.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def ablation_study(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    ablation_groups: dict[str, list[str]],
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    baseline_rmse = cross_validate(model_cls, model_kwargs, X, y, groups, n_splits, seed)["mean"]["rmse"]
    rows = [{"group": "baseline", "rmse": baseline_rmse, "delta_rmse": 0.0}]

    for group_name, cols_to_remove in ablation_groups.items():
        keep = [i for i, f in enumerate(feature_names) if f not in cols_to_remove]
        if not keep:
            continue
        rmse = cross_validate(model_cls, model_kwargs, X[:, keep], y, groups, n_splits, seed)["mean"]["rmse"]
        rows.append({"group": group_name, "rmse": rmse, "delta_rmse": rmse - baseline_rmse})

    return pd.DataFrame(rows)


def hyperparameter_sweep(
    model_cls,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fixed_kwargs: dict | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    from itertools import product as iterproduct

    fixed_kwargs = fixed_kwargs or {}
    keys = list(param_grid.keys())
    rows = []

    for combo in iterproduct(*param_grid.values()):
        kwargs = {**fixed_kwargs, **dict(zip(keys, combo))}
        result = cross_validate(model_cls, kwargs, X, y, groups, n_splits, seed)
        rows.append({**dict(zip(keys, combo)), "rmse": result["mean"]["rmse"], "mae": result["mean"]["mae"]})

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
