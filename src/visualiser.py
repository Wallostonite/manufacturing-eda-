# ================================================================
# src/visualiser.py — 6-chart generator for PrecisionForge EDA
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config import REPORTS_DIR, logger

PLOTS_DIR = REPORTS_DIR / "plots"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


class Visualiser:
    """
    Generates the 6 charts required by the README:
      1 — Defect rate distribution (histogram + KDE)
      2 — Efficiency by plant type (box plot)
      3 — Defect type bar chart
      4 — Time trend of defect_rate_pct
      5 — Maintenance vs defect scatter
      6 — Severity breakdown
    """

    def __init__(self, df: pd.DataFrame, anomaly_df: pd.DataFrame = None,
                 engine_results: dict = None):
        self.df             = df
        self.anomaly_df     = anomaly_df if anomaly_df is not None else pd.DataFrame()
        self.engine_results = engine_results or {}
        PLOTS_DIR.mkdir(exist_ok=True)

    # ── CHART 1 ──────────────────────────────────────────────────────

    def defect_rate_distribution(self) -> pathlib.Path:
        """Histogram + KDE of defect_rate_pct with 5% threshold line."""
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(self.df["defect_rate_pct"], bins=50, kde=True,
                     color="#3498db", ax=ax)
        ax.axvline(5, color="#e74c3c", linewidth=2, linestyle="--",
                   label="Board threshold (5%)")
        over = (self.df["defect_rate_pct"] > 5).mean() * 100
        ax.set_xlabel("Defect Rate (%)", fontsize=12)
        ax.set_ylabel("Production Runs", fontsize=12)
        ax.set_title(
            f"Q1 — Defect Rate Distribution  ({over:.1f}% of runs exceed 5%)",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = PLOTS_DIR / "01_defect_rate_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 2 ──────────────────────────────────────────────────────

    def efficiency_by_plant_type(self) -> pathlib.Path:
        """Box plot: efficiency_pct distribution by plant_type."""
        order = (
            self.df.groupby("plant_type")["efficiency_pct"]
            .median()
            .sort_values(ascending=False)
            .index
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(
            data=self.df, x="plant_type", y="efficiency_pct",
            order=order, palette="Blues_r", ax=ax
        )
        ax.axhline(self.df["efficiency_pct"].mean(), color="#e74c3c",
                   linestyle="--", linewidth=1.5,
                   label=f"Overall mean {self.df['efficiency_pct'].mean():.1f}%")
        ax.set_xlabel("Plant Type", fontsize=12)
        ax.set_ylabel("Efficiency (%)", fontsize=12)
        ax.set_title("Q1 — Efficiency by Plant Type", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = PLOTS_DIR / "02_efficiency_by_plant_type.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 3 ──────────────────────────────────────────────────────

    def defect_type_bar(self) -> pathlib.Path:
        """Horizontal bar: count of each primary_defect_type."""
        counts = (
            self.df["primary_defect_type"]
            .dropna()
            .value_counts()
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        palette = sns.color_palette("Reds_r", len(counts))
        bars = ax.barh(counts.index, counts.values, color=palette, edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax.text(val + 20, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=10)
        ax.set_xlabel("Number of Runs", fontsize=12)
        ax.set_title("Q3 — Defect Type Breakdown", fontsize=13, fontweight="bold")
        ax.set_xlim(0, counts.max() * 1.15)
        plt.tight_layout()
        path = PLOTS_DIR / "03_defect_type_bar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 4 ──────────────────────────────────────────────────────

    def defect_rate_trend(self) -> pathlib.Path:
        """Line chart: monthly mean defect_rate_pct with forecast overlay."""
        work = self.df.copy()
        work["_month"] = (
            pd.to_datetime(work["run_date"], errors="coerce")
            .dt.to_period("M")
            .astype(str)
        )
        monthly = (
            work.groupby("_month")["defect_rate_pct"]
            .mean()
            .reset_index()
            .sort_values("_month")
        )
        monthly["rolling_3m"] = monthly["defect_rate_pct"].rolling(3).mean()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly["_month"], monthly["defect_rate_pct"],
                marker="o", markersize=4, linewidth=1.5,
                color="#3498db", label="Monthly mean")
        ax.plot(monthly["_month"], monthly["rolling_3m"],
                linewidth=2.5, color="#2ecc71", linestyle="--", label="3-month rolling avg")
        ax.axhline(5, color="#e74c3c", linestyle=":", linewidth=1.5,
                   label="Board threshold (5%)")

        # Overlay forecast if available
        fc_results = self.engine_results.get("forecast", {}).get("defect_rate_pct")
        if fc_results:
            next_val = fc_results["next_period"]
            x_pos    = len(monthly["_month"]) - 0.5
            ax.annotate(
                f"Forecast: {next_val:.3f}",
                xy=(x_pos, next_val),
                xytext=(x_pos - 1.5, next_val + 0.3),
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                fontsize=10, color="#e74c3c",
            )

        n = max(1, len(monthly) // 8)
        ax.set_xticks(monthly["_month"][::n])
        ax.set_xticklabels(monthly["_month"][::n], rotation=45, ha="right")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Mean Defect Rate (%)", fontsize=12)
        ax.set_title("Q4 — Monthly Defect Rate Trend", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = PLOTS_DIR / "04_defect_rate_trend.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 5 ──────────────────────────────────────────────────────

    def downtime_vs_defect(self) -> pathlib.Path:
        """Scatter: downtime_mins vs defect_rate_pct coloured by equipment_status."""
        sample = self.df.sample(min(3000, len(self.df)), random_state=42)
        palette = {"Operational": "#3498db", "Maintenance": "#e74c3c"}

        fig, ax = plt.subplots(figsize=(9, 6))
        for status, grp in sample.groupby("equipment_status"):
            ax.scatter(
                grp["downtime_mins"], grp["defect_rate_pct"],
                alpha=0.35, s=12,
                color=palette.get(status, "#95a5a6"),
                label=status
            )
        ax.axhline(5, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.7,
                   label="5% threshold")
        ax.set_xlabel("Downtime (mins)", fontsize=12)
        ax.set_ylabel("Defect Rate (%)", fontsize=12)
        ax.set_title("Q3 — Downtime vs Defect Rate by Equipment Status",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = PLOTS_DIR / "05_downtime_vs_defect.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 6 ──────────────────────────────────────────────────────

    def severity_breakdown(self) -> pathlib.Path:
        """Stacked bar: severity breakdown by plant_type."""
        order = ["Critical", "Major", "Minor", "None"]
        avail = [s for s in order if s in self.df["primary_severity"].dropna().unique()]

        pivot = (
            self.df.groupby(["plant_type", "primary_severity"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=avail, fill_value=0)
        )
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pivot_pct = pivot_pct.sort_values("Critical", ascending=False)

        colors = {"Critical": "#e74c3c", "Major": "#f39c12",
                  "Minor": "#f1c40f", "None": "#95a5a6"}

        fig, ax = plt.subplots(figsize=(11, 6))
        bottom = np.zeros(len(pivot_pct))
        for sev in avail:
            vals = pivot_pct[sev].values
            ax.bar(pivot_pct.index, vals, bottom=bottom,
                   color=colors.get(sev, "#bdc3c7"),
                   label=sev, edgecolor="white", width=0.6)
            bottom += vals

        ax.set_ylabel("Share of Runs (%)", fontsize=12)
        ax.set_xlabel("Plant Type", fontsize=12)
        ax.set_title("Q3 — Severity Breakdown by Plant Type",
                     fontsize=13, fontweight="bold")
        ax.legend(title="Severity", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
        plt.tight_layout()
        path = PLOTS_DIR / "06_severity_breakdown.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── RUN ALL ───────────────────────────────────────────────────────

    def run_all(self) -> list:
        paths = [
            self.defect_rate_distribution(),
            self.efficiency_by_plant_type(),
            self.defect_type_bar(),
            self.defect_rate_trend(),
            self.downtime_vs_defect(),
            self.severity_breakdown(),
        ]
        logger.info(f"[PLOT] All {len(paths)} charts saved → {PLOTS_DIR}")
        return paths
