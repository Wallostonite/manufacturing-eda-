# PrecisionForge Manufacturing — EDA Findings
**Prepared for: Operations Board**
**Dataset: 38,430 production runs | Manufacturing Data**

---

## Executive Summary

Three findings matter before this board meeting. One is a warning, one clears a misconception, and one reveals where the real problem is hiding.

**1. The defect rate forecast is worsening — next period estimated at 2.41%.**
The current mean sits at 2.41%, well below the board's 5% threshold. But the exponential smoothing model projects the trend is heading in the wrong direction. The rate has not crossed 5% yet, but it has been climbing. If the pattern continues, the board threshold becomes relevant faster than current averages suggest.

**2. Plant type and shift have no statistically significant effect on defect rates.**
Both ANOVA and Kruskal-Wallis tests confirm this. The difference between the best plant (Precision, 2.47%) and the worst (Food, 2.35%) is 0.12 percentage points — noise, not signal. Reallocating production across plants or rebalancing shift schedules will not move the defect rate. The cause lies elsewhere.

**3. Critical-severity runs are 2.8× worse on defect rate — and they account for 13% of all production.**
This is where the problem is concentrated. Critical runs average a 5.44% defect rate, already above the board threshold. Non-critical runs average 1.94%. The priority is not improving the average; it is preventing Critical-severity runs from occurring.

---

## Finding 1 — Defect Rate by Plant Type and Shift (Q1)

### Does plant type explain the defect rate?

| Rank | Plant Type     | Mean Defect Rate | Median | Runs  |
|------|----------------|-----------------|--------|-------|
| 1    | Precision      | 2.469%          | 0.000% | 3,889 |
| 2    | Chemical       | 2.453%          | 0.000% | 3,798 |
| 3    | Advanced       | 2.434%          | 0.000% | 3,869 |
| 4    | Electronics    | 2.430%          | 0.000% | 3,817 |
| 5    | Pharmaceutical | 2.427%          | 0.000% | 3,923 |
| 6    | Automotive     | 2.382%          | 0.000% | 7,577 |
| 7    | Steel          | 2.374%          | 0.000% | 3,873 |
| 8    | Food           | 2.352%          | 0.000% | 3,884 |

**ANOVA result: F = 1.040, p = 0.403 — not significant.**
**Kruskal-Wallis result: H = 7.567, p = 0.477 — not significant.**

The spread from best to worst plant is 0.117 percentage points. Two independent statistical tests — one assuming normality, one not — both confirm this difference is within expected random variation. The board should not treat any plant type as a defect problem or a model of excellence based on these numbers alone.

The median defect rate of 0.000% across all plant types indicates that the majority of runs produce zero defects. The mean is pulled up by the subset of runs that go wrong — those Critical-severity events (see Finding 3).

### Does shift explain the defect rate?

| Rank | Shift     | Mean Defect Rate | Runs   |
|------|-----------|-----------------|--------|
| 1    | Afternoon | 2.439%          | 12,926 |
| 2    | Night     | 2.404%          | 12,640 |
| 3    | Morning   | 2.362%          | 12,864 |

**ANOVA result: F = 1.975, p = 0.139 — not significant.**
**Kruskal-Wallis result: H = 3.424, p = 0.181 — not significant.**

Afternoon is marginally higher than Morning by 0.077 percentage points. This is not statistically meaningful. Adjusting shift patterns, moving experienced staff to Afternoon, or attributing defect responsibility to shift managers based on this data would not be justified.

---

## Finding 2 — What Predicts Defect Rate? (Q2)

Feature importance was measured by Pearson correlation between each numeric input and `defect_rate_pct`. Higher absolute correlation means stronger predictive value.

| Feature                  | Correlation (|r|) | Direction |
|--------------------------|------------------|-----------|
| efficiency_pct           | 0.0152           | positive  |
| yield_pct                | 0.0132           | positive  |
| downtime_mins            | 0.0075           | positive  |

**The honest read: no measured feature predicts the defect rate.**

An absolute correlation below 0.02 means that knowing a run's efficiency, yield, or downtime tells you almost nothing about whether it will produce defects. The maximum variance explained by any single feature is less than 0.03% (r² < 0.0003).

This is a significant finding. It means:
- Defects are not a linear function of any operational metric currently being recorded
- The causes of Critical-severity runs are likely discrete events (equipment failure, material batch issue, operator error) rather than gradual degradation captured in continuous metrics
- Predictive modelling using these features alone will not be viable without additional data

**One strong correlation that does exist:** `efficiency_pct` and `yield_pct` have a correlation of r = 0.972. These two metrics move together almost perfectly — recording both adds no additional information. One can be dropped from dashboards without losing any analytical value.

---

## Finding 3 — Equipment Status and Severity (Q3)

### Critical runs: a factory within the factory

| Metric                | Critical Runs | Non-Critical Runs | Difference |
|-----------------------|--------------|-------------------|------------|
| Run count             | 5,084        | 33,346            | —          |
| Share of all runs     | 13.23%       | 86.77%            | —          |
| Mean defect rate      | 5.44%        | 1.94%             | **+3.50 pp** |
| Mean downtime (mins)  | 60.93        | 59.53             | +1.40 mins |
| Mean efficiency       | 80.01%       | 79.94%            | +0.07 pp   |

Critical runs produce a mean defect rate of **5.44%** — already above the board threshold of 5%. Non-critical runs average 1.94%. The ratio is **2.8×**.

What the downtime and efficiency numbers tell us: Critical runs have only marginally higher downtime (+1.4 minutes) and negligibly different efficiency (+0.07%). This rules out "the machine was breaking down" as a general explanation. Critical runs are not distinguishable from normal runs by the continuous metrics captured — they appear normal until they are not.

This reinforces the conclusion from Finding 2: defect events are discrete, not gradual. Something triggers them — a specific batch, a tool change, a process deviation — and it is not being captured in the current data schema.

### Does downtime correlate with defect rate?

Pearson correlation between downtime and defect rate: **r = 0.0075** — effectively zero.

A machine sitting idle for longer does not produce more defects when it resumes. Preventive maintenance triggered purely by downtime thresholds is unlikely to reduce the defect rate.

---

## Finding 4 — Defect Rate Forecast (Q4)

### Is the defect rate improving or worsening?

| Month   | Mean Defect Rate | Month-on-Month | 3-Month Rolling Avg |
|---------|-----------------|----------------|-------------------|
| Jul-24  | 2.407%          | −4.4%          | 2.416%            |
| Aug-24  | 2.479%          | +3.0%          | 2.468%            |
| Sep-24  | 2.368%          | −4.5%          | 2.418%            |
| Oct-24  | 2.519%          | +6.4%          | 2.455%            |
| Nov-24  | 2.471%          | −1.9%          | 2.453%            |
| Dec-24  | 2.318%          | −6.2%          | 2.436%            |

**Exponential smoothing forecast — next period: 2.409% (trend: WORSENING)**

The month-on-month figures are noisy — defect rate fluctuates up and down without a clean directional signal. The three-month rolling average has been drifting upward through the second half of 2024 before a partial pullback in December. The smoothing model, which weights recent observations more heavily, projects the next period will rise to 2.409%.

The board's 5% threshold is not in immediate danger at current levels. However, the model is flagging a worsening direction. The distance between the current mean (2.41%) and the threshold (5.00%) is 2.59 percentage points — a 108% increase required to breach it. The concern is not the absolute level but the trajectory, especially given that Critical runs already breach the threshold every time they occur.

---

## Recommended Actions

| Priority | Action | Owner |
|----------|--------|-------|
| High | Investigate the root causes of Critical-severity runs — pull the 5,084 flagged records from `reports/anomalies.csv` and identify whether they cluster by equipment ID, operator, material batch, or time of year | Quality / Operations |
| High | Extend the data schema to capture discrete event data: equipment faults, material batch IDs, operator IDs, tool change timestamps — current continuous metrics cannot predict defects | Data Engineering |
| Medium | Set up a monitoring alert when the 3-month rolling defect rate exceeds 2.6% — this gives a warning buffer before approaching the 5% board threshold | Analytics / Operations |
| Medium | Consolidate `efficiency_pct` and `yield_pct` into a single metric on operational dashboards — correlation r = 0.972 means they contain the same information | Analytics |
| Low | Do not adjust plant-type staffing or shift schedules to manage defect rates — statistical tests confirm these factors are not driving the variation | Operations Management |

---

## Data Quality Notes

- **38,430 production runs** across 8 plant types and 3 shifts
- **3.22% of cells** contain missing values (45,800 cells) — primarily in supplementary fields; core metrics are complete
- **0 duplicate rows** — each run ID appears once
- Statistical thresholds: ANOVA and Kruskal-Wallis significance at p < 0.05; IQR fence at Q1 − 1.5×IQR / Q3 + 1.5×IQR; Z-score threshold at |z| > 3.0
- Full outputs: `reports/analysis_report.txt`, `reports/manufacturing_eda_report.html`, `reports/anomalies.csv`, `reports/plots/`

---

*Analysis produced by the Junior Data Analyst team using the PrecisionForge EDA pipeline (Module 06). Charts available in `reports/plots/`. HTML report at `reports/manufacturing_eda_report.html`. Raw findings reproducible by running `python run.py`.*
