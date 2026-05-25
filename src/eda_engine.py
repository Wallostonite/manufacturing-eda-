# ================================================================
# src/eda_engine.py — ManufacturingEDAEngine
# ================================================================
# CONTEXT:
#   Analyses PrecisionForge production-run data to answer four
#   board-level questions:
#     Q1 — Does defect_rate_pct differ significantly by plant_type / shift?
#     Q2 — Which features most strongly predict defect_rate_pct?
#     Q3 — Does equipment_status correlate with defect severity?
#     Q4 — What is the forecast for defect_rate_pct next period?
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from config import (
    INDUSTRY,
    DATA_PATH,
    REPORTS_DIR,
    TOP_N_GROUPS,
    CORRELATION_THRESHOLD,
    GROUP_COLS,
    VALUE_COL,
    TARGET_COL,
    CORR_COLS,
    EXCLUDE_COLS,
    logger,
)


class EDAEngine:
    """
    ManufacturingEDAEngine — EDA for PrecisionForge production-run data.

    Standard methods (inherited pattern):
      1. load()                          → loads CSV, excludes metadata/ID cols
      2. profile()                       → shape, nulls, descriptive stats
      3. group_analysis(group, value)    → defect_rate_pct by plant_type / shift
      4. correlation(col_a, col_b)       → Pearson r between numeric features
      5. time_trends()                   → monthly defect_rate_pct trend
      6. report(save)                    → prints + saves analysis_report.txt

    Manufacturing-specific methods:
      7. anova_test(group_col, value_col)    → one-way ANOVA (F-stat + p-value)
      8. kruskal_test(group_col, value_col)  → non-parametric alternative
      9. feature_importance(target_col)      → ranks features by |r| with target
     10. forecast_next_period(col)           → exponential smoothing via statsmodels
    """

    def __init__(self):
        self.df       = None
        self.results  = {}
        self.num_cols = []
        self.cat_cols = []
        self._status  = "ready"
        logger.info(f"EDAEngine initialised — industry: {INDUSTRY}")

    # ── STANDARD METHODS ─────────────────────────────────────────────

    def load(self) -> "EDAEngine":
        """Load CSV, strip pipeline metadata, exclude ID/constant columns."""
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"processed-data.csv not found at: {DATA_PATH}"
            )

        logger.info(f"[EDA] Loading: {DATA_PATH.name}")
        self.df = pd.read_csv(DATA_PATH, low_memory=False)
        logger.info(f"[EDA] Loaded {len(self.df):,} rows × {self.df.shape[1]} columns")

        # Strip pipeline metadata (columns starting with _)
        analysis_df = self.df.drop(
            columns=[c for c in self.df.columns if c.startswith("_")],
            errors="ignore",
        )

        # Exclude known non-metric columns
        self.num_cols = [
            c for c in analysis_df.select_dtypes(include=["number"]).columns
            if c not in EXCLUDE_COLS
        ]
        self.cat_cols = analysis_df.select_dtypes(include=["object"]).columns.tolist()

        self._status = "loaded"
        logger.info(
            f"[EDA] Column types: {len(self.num_cols)} numeric, "
            f"{len(self.cat_cols)} categorical"
        )
        return self

    def profile(self) -> "EDAEngine":
        """Compute shape, null rates, descriptive stats, and categorical profiles."""
        logger.info("[EDA] Computing dataset profile...")

        profile = {
            "rows":             len(self.df),
            "columns":          len(self.df.columns),
            "numeric_cols":     len(self.num_cols),
            "categorical_cols": len(self.cat_cols),
            "total_nulls":      int(self.df.isna().sum().sum()),
            "null_pct":         round(self.df.isna().sum().sum() / self.df.size * 100, 2),
            "memory_mb":        round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            "duplicates":       int(self.df.duplicated().sum()),
        }

        if self.num_cols:
            desc = self.df[self.num_cols].describe().round(3)
            profile["descriptive_stats"] = desc.to_dict(orient="index")

        cat_profiles = {}
        for col in self.cat_cols[:8]:
            vc = self.df[col].value_counts()
            cat_profiles[col] = {
                "unique_count": int(self.df[col].nunique()),
                "top_5":        vc.head(5).to_dict(),
                "null_count":   int(self.df[col].isna().sum()),
            }
        profile["categorical_profiles"] = cat_profiles

        self.results["profile"] = profile
        logger.info(
            f"[EDA] Profile complete — {profile['rows']:,} rows | {profile['null_pct']}% nulls"
        )
        return self

    def group_analysis(self, group_col: str = None, value_col: str = None) -> "EDAEngine":
        """
        Compute ranked aggregates of value_col by group_col.

        Defaults to GROUP_COLS × VALUE_COL (plant_type/shift vs defect_rate_pct).
        """
        logger.info("[EDA] Running group analysis...")

        if group_col and value_col:
            pairs = [(group_col, value_col)]
        else:
            pairs = [
                (g, VALUE_COL) for g in GROUP_COLS
                if g in self.df.columns and VALUE_COL in self.df.columns
            ]
            if not pairs and self.cat_cols and self.num_cols:
                pairs = [(self.cat_cols[0], self.num_cols[0])]

        group_results = {}
        for cat, num in pairs:
            if cat not in self.df.columns or num not in self.df.columns:
                logger.warning(f"[EDA] Skipping — missing column: {cat!r} or {num!r}")
                continue

            agg = (
                self.df.groupby(cat)[num]
                .agg(["count", "mean", "median", "std", "min", "max"])
                .round(3)
                .reset_index()
            )
            agg["rank"] = agg["mean"].rank(ascending=False).astype(int)
            group_results[cat] = {
                num: agg.sort_values("rank").head(TOP_N_GROUPS).to_dict(orient="records")
            }

        self.results["group_analysis"] = group_results
        logger.info(f"[EDA] Group analysis: {len(group_results)} grouping variable(s)")
        return self

    def correlation(self, col_a: str = None, col_b: str = None) -> "EDAEngine":
        """
        Pearson correlation between numeric columns.

        Defaults to CORR_COLS (defect_rate_pct, efficiency_pct, downtime_mins,
        yield_pct, days_since_maintenance). Pass col_a + col_b for a single pair.
        """
        logger.info("[EDA] Computing correlation...")

        if col_a and col_b:
            analysis_cols = [c for c in [col_a, col_b] if c in self.df.columns]
        else:
            analysis_cols = [c for c in CORR_COLS if c in self.df.columns]
            if len(analysis_cols) < 2:
                analysis_cols = self.num_cols

        if len(analysis_cols) < 2:
            logger.warning("[EDA] Not enough numeric columns for correlation")
            return self

        # Drop zero-variance columns (like days_since_maintenance = all 0s)
        valid_cols = [c for c in analysis_cols if self.df[c].std() > 0]
        if len(valid_cols) < 2:
            logger.warning("[EDA] Not enough variance in columns for correlation")
            self.results["correlation"] = {"matrix": {}, "strong_pairs": []}
            return self

        corr_matrix = self.df[valid_cols].corr(numeric_only=True).round(3)

        corr_pairs = []
        for i, ca in enumerate(valid_cols):
            for j, cb in enumerate(valid_cols):
                if i >= j:
                    continue
                val = corr_matrix.loc[ca, cb]
                if abs(val) < CORRELATION_THRESHOLD:
                    continue
                strength = (
                    "STRONG"   if abs(val) > 0.7 else
                    "MODERATE" if abs(val) > 0.5 else
                    "WEAK"
                )
                corr_pairs.append({
                    "col_a":       ca,
                    "col_b":       cb,
                    "correlation": float(val),
                    "strength":    strength,
                    "direction":   "positive" if val > 0 else "negative",
                })

        corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        self.results["correlation"] = {
            "matrix":       corr_matrix.to_dict(),
            "strong_pairs": corr_pairs[:10],
        }
        logger.info(
            f"[EDA] Correlation: {len(corr_pairs)} meaningful pairs "
            f"(threshold |r| > {CORRELATION_THRESHOLD})"
        )
        return self

    def time_trends(self) -> "EDAEngine":
        """Monthly trend for defect_rate_pct derived from run_date."""
        time_keywords = ["month", "period", "quarter", "year", "date"]
        time_cols = [
            c for c in self.df.columns
            if any(kw in c.lower() for kw in time_keywords)
        ]
        if not time_cols:
            logger.info("[EDA] No time column found — skipping time trends")
            return self

        time_col  = time_cols[0]
        work_df   = self.df.copy()
        group_col = time_col

        if "date" in time_col.lower():
            work_df["_month"] = (
                pd.to_datetime(work_df[time_col], errors="coerce")
                .dt.to_period("M")
                .astype(str)
            )
            group_col = "_month"
        elif work_df[time_col].nunique() > 200:
            logger.info(f"[EDA] {time_col} has too many unique values — skipping")
            return self

        logger.info(f"[EDA] Computing time trends on '{time_col}' (grouped by month)...")

        trend_targets = [c for c in [VALUE_COL] if c in work_df.columns] or self.num_cols[:1]
        trend_results = {}

        for num_col in trend_targets:
            if num_col not in work_df.columns:
                continue
            monthly = (
                work_df.groupby(group_col)[num_col]
                .agg(["count", "sum", "mean"])
                .round(3)
                .reset_index()
                .sort_values(group_col)
            )
            monthly["mom_change"] = monthly["mean"].diff()
            monthly["mom_pct"]    = monthly["mean"].pct_change().mul(100).round(1)
            monthly["rolling_3m"] = monthly["mean"].rolling(3).mean().round(3)
            trend_results[num_col] = monthly.to_dict(orient="records")

        self.results["time_trends"] = trend_results
        logger.info(f"[EDA] Time trends: {len(trend_results)} metrics analysed")
        return self

    def report(self, save: bool = True) -> "EDAEngine":
        """Print and save analysis_report.txt."""
        lines = [
            "═" * 65,
            f"  PRECISIONFORGE — EDA REPORT  |  INDUSTRY: {INDUSTRY.upper()}",
            "═" * 65,
        ]

        if "profile" in self.results:
            p = self.results["profile"]
            lines += [
                "",
                "  DATASET OVERVIEW",
                f"    Records:              {p['rows']:,}",
                f"    Columns:              {p['columns']}",
                f"    Numeric columns:      {p['numeric_cols']}",
                f"    Categorical columns:  {p['categorical_cols']}",
                f"    Missing values:       {p['total_nulls']:,} ({p['null_pct']}%)",
                f"    Duplicate rows:       {p['duplicates']:,}",
                f"    Memory usage:         {p['memory_mb']} MB",
            ]
            if "descriptive_stats" in p:
                lines += ["", "  DESCRIPTIVE STATISTICS"]
                lines.append(
                    f"    {'Metric':<30} {'Mean':>10} {'Median':>10} {'Std Dev':>10}"
                )
                lines.append("    " + "-" * 63)
                for stat, cols in p["descriptive_stats"].items():
                    if stat != "mean":
                        continue
                    for col, mean_val in list(cols.items())[:8]:
                        med = p["descriptive_stats"].get("50%", {}).get(col)
                        std = p["descriptive_stats"].get("std", {}).get(col)
                        lines.append(
                            f"    {col:<30} {mean_val:>10.3f} "
                            f"{(med or 0):>10.3f} {(std or 0):>10.3f}"
                        )

        if "group_analysis" in self.results:
            lines += ["", "  GROUP ANALYSIS"]
            for cat_col, metrics in self.results["group_analysis"].items():
                for num_col, rows in metrics.items():
                    lines += [
                        "",
                        f"    {num_col.upper()} BY {cat_col.upper()}",
                        f"    {'Group':<28} {'Mean':>10} {'Median':>10} "
                        f"{'Count':>8} {'Rank':>6}",
                        "    " + "-" * 65,
                    ]
                    for row in rows[:TOP_N_GROUPS]:
                        g  = str(row.get(cat_col, ""))[:27]
                        lines.append(
                            f"    {g:<28} {row.get('mean', 0):>10.3f} "
                            f"{row.get('median', 0):>10.3f} "
                            f"{row.get('count', 0):>8,} {row.get('rank', 0):>6}"
                        )

        if "correlation" in self.results:
            pairs = self.results["correlation"].get("strong_pairs", [])
            if pairs:
                lines += [
                    "",
                    f"  TOP CORRELATIONS (|r| > {CORRELATION_THRESHOLD})",
                    f"    {'Column A':<28} {'Column B':<28} {'r':>8}  Strength",
                    "    " + "-" * 70,
                ]
                for p in pairs[:8]:
                    lines.append(
                        f"    {p['col_a']:<28} {p['col_b']:<28} "
                        f"{p['correlation']:>8.3f}  {p['strength']} {p['direction']}"
                    )
            else:
                lines += ["", "  CORRELATIONS: No pairs exceed threshold "
                          f"(|r| > {CORRELATION_THRESHOLD})"]

        if "anova" in self.results:
            lines += ["", "  ANOVA TESTS"]
            for key, res in self.results["anova"].items():
                sig = "SIGNIFICANT" if res["p_value"] < 0.05 else "not significant"
                lines.append(
                    f"    {key:<40} F={res['f_stat']:.3f}  "
                    f"p={res['p_value']:.4f}  [{sig}]"
                )

        if "kruskal" in self.results:
            lines += ["", "  KRUSKAL-WALLIS TESTS"]
            for key, res in self.results["kruskal"].items():
                sig = "SIGNIFICANT" if res["p_value"] < 0.05 else "not significant"
                lines.append(
                    f"    {key:<40} H={res['h_stat']:.3f}  "
                    f"p={res['p_value']:.4f}  [{sig}]"
                )

        if "feature_importance" in self.results:
            lines += ["", "  FEATURE IMPORTANCE (|r| with defect_rate_pct)"]
            lines.append(f"    {'Feature':<30} {'|r|':>8}  Direction")
            lines.append("    " + "-" * 50)
            for feat in self.results["feature_importance"]:
                lines.append(
                    f"    {feat['feature']:<30} {feat['abs_r']:>8.4f}  "
                    f"{feat['direction']}"
                )

        if "forecast" in self.results:
            lines += ["", "  FORECAST (exponential smoothing)"]
            for col, res in self.results["forecast"].items():
                lines.append(
                    f"    {col}: next period estimate = {res['next_period']:.3f}  "
                    f"(trend: {res['trend']})"
                )

        if "time_trends" in self.results:
            lines += ["", "  TIME TRENDS (last 6 periods)"]
            for metric, rows in self.results["time_trends"].items():
                if not rows:
                    continue
                period_key = list(rows[0].keys())[0]
                lines += [
                    "",
                    f"    {metric.upper()}",
                    f"    {'Period':<12} {'Mean':>10} {'MoM %':>9} {'Rolling 3':>12}",
                    "    " + "-" * 46,
                ]
                for row in rows[-6:]:
                    mom_p   = row.get("mom_pct")   or 0.0
                    rolling = row.get("rolling_3m") or 0.0
                    lines.append(
                        f"    {str(row.get(period_key, '')):<12} "
                        f"{row.get('mean', 0):>10.3f} "
                        f"{mom_p:>8.1f}% {rolling:>12.3f}"
                    )

        lines += [
            "",
            "  NEXT STEPS:",
            "    → Investigate Critical-severity runs for root-cause patterns",
            "    → Use feature_importance rankings for ML feature selection (Module 09)",
            "    → Monitor forecast trend — escalate if defect_rate_pct rises",
            "",
            "═" * 65,
        ]

        report_text = "\n".join(lines)
        print(report_text)

        if save:
            path = REPORTS_DIR / "analysis_report.txt"
            path.write_text(report_text, encoding="utf-8")
            logger.info(f"[EDA] Report saved: {path}")

        return self

    # ── MANUFACTURING-SPECIFIC METHODS ────────────────────────────────

    def anova_test(self, group_col: str = None, value_col: str = None) -> "EDAEngine":
        """
        One-way ANOVA: tests whether group means differ significantly.

        F-statistic = variance between groups / variance within groups
        p < 0.05 → reject null hypothesis (group means are NOT all equal)

        Args:
            group_col   categorical grouping column (default: plant_type, then shift)
            value_col   numeric metric to test (default: defect_rate_pct)
        """
        val = value_col or VALUE_COL
        groups_to_test = [(group_col, val)] if group_col else [(g, val) for g in GROUP_COLS]

        anova_results = {}
        for grp, metric in groups_to_test:
            if grp not in self.df.columns or metric not in self.df.columns:
                logger.warning(f"[ANOVA] Column missing: {grp!r} or {metric!r}")
                continue

            group_arrays = [
                grp_df[metric].dropna().values
                for _, grp_df in self.df.groupby(grp)
                if len(grp_df[metric].dropna()) > 1
            ]
            if len(group_arrays) < 2:
                continue

            f_stat, p_value = scipy_stats.f_oneway(*group_arrays)
            key = f"{metric} ~ {grp}"
            anova_results[key] = {
                "group_col": grp,
                "value_col": metric,
                "f_stat":    round(float(f_stat), 4),
                "p_value":   round(float(p_value), 6),
                "significant": p_value < 0.05,
            }
            logger.info(
                f"[ANOVA] {key}: F={f_stat:.3f}  p={p_value:.4f}  "
                f"{'SIGNIFICANT' if p_value < 0.05 else 'not significant'}"
            )

        self.results["anova"] = anova_results
        return self

    def kruskal_test(self, group_col: str = None, value_col: str = None) -> "EDAEngine":
        """
        Kruskal-Wallis test: non-parametric alternative to one-way ANOVA.

        Use when defect_rate_pct is skewed (many zeros in non-defective runs).
        H-statistic and p-value interpretation is identical to ANOVA.

        Args:
            group_col   categorical grouping column (default: plant_type, then shift)
            value_col   numeric metric to test (default: defect_rate_pct)
        """
        val = value_col or VALUE_COL
        groups_to_test = [(group_col, val)] if group_col else [(g, val) for g in GROUP_COLS]

        kruskal_results = {}
        for grp, metric in groups_to_test:
            if grp not in self.df.columns or metric not in self.df.columns:
                logger.warning(f"[KRUSKAL] Column missing: {grp!r} or {metric!r}")
                continue

            group_arrays = [
                grp_df[metric].dropna().values
                for _, grp_df in self.df.groupby(grp)
                if len(grp_df[metric].dropna()) > 1
            ]
            if len(group_arrays) < 2:
                continue

            h_stat, p_value = scipy_stats.kruskal(*group_arrays)
            key = f"{metric} ~ {grp}"
            kruskal_results[key] = {
                "group_col": grp,
                "value_col": metric,
                "h_stat":    round(float(h_stat), 4),
                "p_value":   round(float(p_value), 6),
                "significant": p_value < 0.05,
            }
            logger.info(
                f"[KRUSKAL] {key}: H={h_stat:.3f}  p={p_value:.4f}  "
                f"{'SIGNIFICANT' if p_value < 0.05 else 'not significant'}"
            )

        self.results["kruskal"] = kruskal_results
        return self

    def feature_importance(self, target_col: str = None) -> "EDAEngine":
        """
        Rank numeric features by Pearson |r| with the target column.

        This is a fast, model-free proxy for feature importance.
        Higher |r| → stronger linear relationship with defect_rate_pct.

        Args:
            target_col   the column to predict (default: defect_rate_pct)
        """
        target = target_col or TARGET_COL
        if target not in self.df.columns:
            logger.warning(f"[FEAT] Target column not found: {target!r}")
            return self

        # Candidate features: CORR_COLS minus the target, intersected with df columns
        candidates = [
            c for c in CORR_COLS
            if c != target and c in self.df.columns and self.df[c].std() > 0
        ]

        rankings = []
        for feat in candidates:
            r, p = scipy_stats.pearsonr(
                self.df[feat].dropna(),
                self.df[target].loc[self.df[feat].dropna().index],
            )
            rankings.append({
                "feature":   feat,
                "r":         round(float(r), 4),
                "abs_r":     round(abs(float(r)), 4),
                "p_value":   round(float(p), 6),
                "direction": "positive" if r > 0 else "negative",
                "significant": p < 0.05,
            })

        rankings.sort(key=lambda x: x["abs_r"], reverse=True)
        self.results["feature_importance"] = rankings

        logger.info(
            f"[FEAT] Feature importance: {len(rankings)} features ranked "
            f"against '{target}'"
        )
        return self

    def forecast_next_period(self, col: str = None) -> "EDAEngine":
        """
        Exponential smoothing forecast for the next production period.

        Uses Holt-Winters simple exponential smoothing from statsmodels.
        Fits on monthly aggregated means and forecasts one period ahead.

        Args:
            col   numeric column to forecast (default: defect_rate_pct)
        """
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        target = col or VALUE_COL
        if target not in self.df.columns:
            logger.warning(f"[FORECAST] Column not found: {target!r}")
            return self

        # Build monthly series from run_date
        date_col = next(
            (c for c in self.df.columns if "date" in c.lower()), None
        )
        if date_col is None:
            logger.warning("[FORECAST] No date column — cannot build time series")
            return self

        work = self.df.copy()
        work["_month"] = (
            pd.to_datetime(work[date_col], errors="coerce")
            .dt.to_period("M")
        )
        monthly = (
            work.groupby("_month")[target]
            .mean()
            .sort_index()
        )

        if len(monthly) < 3:
            logger.warning("[FORECAST] Too few periods for smoothing (need ≥ 3)")
            return self

        model  = SimpleExpSmoothing(monthly.values).fit(optimized=True)
        fc     = float(model.forecast(1)[0])
        last   = float(monthly.iloc[-1])
        trend  = "improving" if fc < last else "worsening"

        self.results.setdefault("forecast", {})[target] = {
            "periods_used":  len(monthly),
            "last_actual":   round(last, 4),
            "next_period":   round(fc, 4),
            "trend":         trend,
            "monthly_means": monthly.astype(float).round(4).to_dict(),
        }

        logger.info(
            f"[FORECAST] {target}: last={last:.3f}  "
            f"next={fc:.3f}  trend={trend}"
        )
        return self

    def __str__(self) -> str:
        rows = len(self.df) if self.df is not None else 0
        return (
            f"EDAEngine(industry={INDUSTRY!r}, rows={rows:,}, "
            f"analyses={list(self.results.keys())})"
        )

    def __repr__(self) -> str:
        return f"EDAEngine(industry={INDUSTRY!r}, status={self._status!r})"
