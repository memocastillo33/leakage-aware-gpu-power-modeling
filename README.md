# Leakage-Aware GPU Power Modeling

Regression pipeline for predicting accelerator power draw from GPU telemetry, built
around strict leakage controls. Compares a tree ensemble baseline against a neural
network under grouped cross-validation, with permutation importance, ablation studies,
and automated reporting.

## In plain terms

The chips that run video games and AI workloads consume a lot of electricity, and
measuring that draw directly requires special hardware. This project estimates it
instead, using signals the computer already reports for free — how busy the chip is,
how hot it is, how fast it's running.

The hard part wasn't building the estimator. It was making sure the estimator doesn't
cheat. Imagine studying from a practice book that accidentally has the answers printed
in the back: you ace the practice and fail the real exam, because you never actually
learned anything. Models fail the same way when information about the test set leaks
into training. Everything here is designed so the model is tested only on hardware it
has genuinely never seen — and so that claim can be verified rather than assumed.

## Motivation

Direct power measurement requires dedicated instrumentation. Telemetry counters
(utilization, temperature, clock frequency) are cheap to collect and already exposed by
the driver stack, so a model mapping telemetry to power draw is useful for capacity
planning and scheduling in datacenter environments.

The harder problem is evaluating such a model honestly. Telemetry is autocorrelated
within a device, and per-device hardware variation means a random train/test split
lets the model memorize device-specific offsets rather than learn the underlying
relationship. This project treats leakage prevention as the primary design constraint.

## Approach

**Grouped cross-validation.** `GroupKFold` partitions by `device_id`, so every
evaluation fold contains only devices absent from training. This is the realistic
deployment scenario: predicting power on hardware the model has not seen.

**Fold-local normalization.** Feature scaling is fit on the training fold and applied
to the validation fold. Fitting a scaler on the full dataset before splitting leaks
validation statistics into training.

**Explicit verification.** `sanity_check()` asserts zero group overlap across every
fold rather than assuming the splitter behaved correctly.

**Per-fold seed offsets.** Seeds are set globally for reproducibility and offset by
fold index, so folds are independent but the run is deterministic.

## Structure

```
gpu-power-modeling/
├── main.py                  # Pipeline entry point
├── configs/config.yaml      # All experiment parameters
├── src/
│   ├── data_loader.py       # Synthetic telemetry generation, CSV I/O
│   ├── features.py          # Feature engineering, LeakageAwareScaler
│   ├── models.py            # RandomForest and PyTorch MLP
│   ├── evaluation.py        # CV, metrics, ablation, permutation importance
│   └── report.py            # Plots and Markdown report generation
├── data/
├── outputs/{figures,reports}
└── notebooks/
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
python main.py
```

Runs on CPU. PyPI ships CPU-only PyTorch wheels on Windows; the MLP is small enough
that this is not a constraint. Python 3.14 requires `torch>=2.10`.

## Features

Raw telemetry: `gpu_util`, `mem_util`, `temp_c`, `clock_mhz`

Derived: `gpu_util_roll_mean`, `mem_util_roll_mean` (per-device rolling means),
`temp_clock_ratio`, `util_product`

Target: `power_w`

Derived features use only within-device or row-local operations, so no global
statistics enter the feature matrix before splitting.

## Configuration

All parameters live in `configs/config.yaml`: dataset size and noise, rolling window,
feature list, fold count, model hyperparameters, ablation groups, and output paths.
Experiments are run by editing the config, not the code.

Ablation groups are organized by signal domain — `utilization`, `thermal`,
`frequency`, `interaction` — so each ablation answers which class of physical signal
carries predictive weight.

## Results

Under grouped CV the baseline reaches R² ≈ 0.67. The residual is dominated by
per-device variation: the synthetic generator assigns each device an independent base
power draw, and because evaluation folds contain unseen devices, that offset is not
recoverable from telemetry alone. A random split would report a substantially higher
score by leaking device identity — the gap between the two is the point of the
grouped protocol.

Ablation ranks `utilization` well above the other signal groups. Thermal and frequency
features are largely redundant once utilization is present, which follows from the
generator: temperature and clock are both driven by utilization.

## Output

Each run writes `outputs/reports/report.md` with per-fold metrics, model comparison
tables, hyperparameter sweep results, and plots for predicted-vs-actual, permutation
importance, ablation deltas, and MLP training curves.

## Notes

- Rolling means include the current row. Appropriate for modeling instantaneous power
  from concurrent telemetry; strict forecasting would require shifting the window.
- The full-data MLP fit in `main.py` exists only to produce a training curve. Its
  metrics are not reported as performance.
