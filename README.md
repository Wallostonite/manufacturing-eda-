# PrecisionForge Manufacturing — Exploratory Data Analysis

**Company:** PrecisionForge Manufacturing Group
**Role:** Lead Data Analyst
**Dataset:** 38,430 production runs across 8 plant types

---

## Project Overview

Defect rates breached the 5% board threshold at several plants, but the root cause was unquantified. This project delivers a statistically rigorous EDA of production run data: testing whether plant type, shift, or equipment condition independently drives defect rates; ranking features by predictive power; profiling Critical-severity runs; and forecasting the next period's defect rate using exponential smoothing.

---

## What Was Built

### `EDAEngine` — Core Analysis with Statistical Extensions

| Method | Description |
|---|---|
| `load()` | Loads `data/processed-data.csv`, excludes run_id and zero-variance columns |
| `profile()` | Shape, null rates, descriptive statistics for all 16 numeric columns |
| `group_analysis()` | Grouped mean / median / std ranked by `defect_rate_pct` per plant type and shift |
| `correlation()` | Pearson correlation matrix; filters NaN and pairs below \|r\| > 0.3 |
| `time_trends()` | Monthly defect rate trend with rolling 3-month average |
| `anova_test()` | One-way ANOVA (`scipy.stats.f_oneway`) testing `defect_rate_pct` across groups |
| `kruskal_test()` | Non-parametric Kruskal-Wallis test — used when normality is in question |
| `feature_importance()` | Ranks all numeric features by Pearson \|r\| with `defect_rate_pct` |
| `forecast_next_period()` | Exponential smoothing (`statsmodels.SimpleExpSmoothing`) on monthly means |
| `report(save=True)` | Prints and saves the full structured analysis report |

### `AnomalyDetector` — Outlier Detection with Critical Run Profiling

| Method | Description |
|---|---|
| `run(columns)` | IQR + Z-score consensus anomaly detection |
| `flag_critical_runs()` | Compares mean defect_rate_pct, downtime_mins, and efficiency_pct between Critical and non-Critical severity runs |
| `save_anomalies()` | Saves confirmed anomaly rows to `reports/anomalies.csv` |
| `summary()` | Per-column anomaly summary DataFrame |

### `HTMLReporter` — Formatted HTML Report

Generates `reports/manufacturing_eda_report.html` with:
- Dataset overview stat grid
- ANOVA and Kruskal-Wallis results with significance badges
- Feature importance bar visualisation
- Forecast result with trend indicator
- Critical runs comparison table

### `Visualiser` — Six-Chart Generator

Charts saved to `reports/plots/`:
1. Defect rate distribution — histogram + KDE with 5% board threshold line
2. Efficiency by plant type — box plot sorted by median
3. Defect type breakdown — horizontal bar chart
4. Monthly defect rate trend — line chart with rolling average and forecast annotation
5. Downtime vs defect rate — scatter coloured by equipment status
6. Severity breakdown by plant type — stacked bar

### Jupyter Notebook

`notebooks/manufacturing_deep_dive.ipynb` — answers all four board questions with inline charts.

---

## Key Findings

| Question | Finding |
|---|---|
| Q1 — Plant type | F=1.040, p=0.403 — not significant. The 0.12 pp spread across plants is noise, not signal |
| Q1 — Shift | F=1.975, p=0.139 — not significant. Afternoon is marginally higher but not reliably so |
| Q2 — Feature importance | All features show \|r\| < 0.02. No continuous metric predicts defect rate — events are discrete, not gradual |
| Q3 — Critical runs | 5,084 runs (13.2%) are Critical; mean defect rate 5.44% vs 1.94% for non-Critical — **2.8× higher**, already above the 5% board threshold |
| Q4 — Forecast | Next period: **2.409%** — trend: **WORSENING** |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run.py

# Run unit tests
python tests/test_eda.py
```

---

## Project Structure

```
manufacturing-eda/
├── data/
│   └── processed-data.csv            # 38,430 production runs
├── reports/
│   ├── analysis_report.txt           # Full EDA text report
│   ├── manufacturing_eda_report.html # Styled HTML report
│   ├── anomalies.csv                 # Flagged production runs
│   └── plots/                        # 6 PNG charts
├── notebooks/
│   └── manufacturing_deep_dive.ipynb # Board-level Q1–Q4 analysis
├── src/
│   ├── eda_engine.py                 # EDAEngine with ANOVA, Kruskal, forecast
│   ├── anomaly_detector.py           # AnomalyDetector with flag_critical_runs
│   ├── html_reporter.py              # HTMLReporter class
│   └── visualiser.py                 # Visualiser — 6 charts
├── tests/
│   └── test_eda.py                   # 23 unit tests
├── config.py
└── run.py                            # Pipeline entry point
```

---

## Dataset Reference

| Column | Type | Description |
|---|---|---|
| `run_id` | int | Unique production run identifier |
| `run_date` | date | Date of production run |
| `shift` | category | Morning / Afternoon / Night |
| `plant_type` | category | Automotive / Pharmaceutical / Precision / Food / Steel / Advanced |
| `planned_units` | int | Units scheduled for the run |
| `actual_units` | int | Units actually produced |
| `defective_units` | int | Units that failed QA |
| `efficiency_pct` | float | Actual vs planned output ratio (0–100) |
| `downtime_mins` | float | Unplanned downtime in minutes |
| `yield_pct` | float | Non-defective output as % of actual (0–100) |
| `defect_rate_pct` | float | Defective units as % of actual (0–10) |
| `primary_defect_type` | category | Assembly Error / Dimensional / Electrical Fault / Material Defect / Surface Finish / None |
| `primary_severity` | category | Critical / Major / Minor / None |
| `equipment_type` | category | CNC Machine / Conveyor Belt / Packaging Machine / Welding Robot |
| `equipment_status` | category | Operational / Maintenance |
| `days_since_maintenance` | float | Days since last equipment service |

---

## Tests

23 unit tests covering profile, group analysis, correlation, ANOVA/Kruskal, feature importance, anomaly detection, and critical run profiling.

```bash
python tests/test_eda.py
# All tests passed ✓
```

---

## Outputs

| File | Description |
|---|---|
| `reports/analysis_report.txt` | Full structured EDA text report |
| `reports/manufacturing_eda_report.html` | Styled HTML report with significance badges |
| `reports/anomalies.csv` | Flagged production runs |
| `reports/plots/` | 6 PNG charts |
| `notebooks/manufacturing_deep_dive.ipynb` | Interactive board-level notebook |
| `BUSINESS.md` | Board-ready presentation of findings |
