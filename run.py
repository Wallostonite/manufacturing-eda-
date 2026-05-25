# ================================================================
# run.py — PrecisionForge Manufacturing EDA Entry Point
# ================================================================
# HOW TO RUN:
#   python run.py
#
# WHAT HAPPENS:
#   1. Load processed-data.csv
#   2. Profile the dataset
#   3. Group analysis (defect_rate_pct by plant_type and shift)
#   4. Correlation (numeric feature relationships)
#   5. Time trends (monthly defect_rate_pct)
#   6. ANOVA + Kruskal-Wallis (statistical significance tests)
#   7. Feature importance (ranked predictors of defect_rate_pct)
#   8. Exponential smoothing forecast
#   9. Save analysis_report.txt
#  10. Anomaly detection (IQR + Z-score) + flag_critical_runs()
#  11. Generate HTML report
#  12. Save charts
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import INDUSTRY, logger
from src.eda_engine    import EDAEngine
from src.anomaly_detector import AnomalyDetector
from src.html_reporter import HTMLReporter


def main() -> None:
    logger.info("=" * 60)
    logger.info("  PRECISIONFORGE — MANUFACTURING EDA")
    logger.info(f"  Industry: {INDUSTRY}")
    logger.info("=" * 60)

    # ── PART 1: EDA ENGINE ────────────────────────────────────────
    engine = EDAEngine()
    logger.info(f"Created: {engine}")

    (
        engine
        .load()
        .profile()
        .group_analysis()
        .correlation()
        .time_trends()
        .anova_test()
        .kruskal_test()
        .feature_importance()
        .forecast_next_period()
        .report(save=True)
    )

    logger.info(f"EDA complete: {engine}")

    # ── PART 2: ANOMALY DETECTION ─────────────────────────────────
    if engine.df is not None:
        logger.info("")
        logger.info("[ANOMALY] Starting anomaly detection...")

        detector = AnomalyDetector(engine.df)
        detector.run()
        detector.flag_critical_runs()
        detector.save_anomalies()

        print()
        print("  ANOMALY DETECTION SUMMARY:")
        print("  " + "-" * 50)
        print(detector.summary().to_string(index=False))
        print()
        print(f"  Confirmed anomaly rows: {len(detector.confirmed):,}")
        print(f"  Saved to: reports/anomalies.csv")

        if "critical_runs" in detector.results:
            cr = detector.results["critical_runs"]
            print()
            print("  CRITICAL RUNS PROFILE:")
            print("  " + "-" * 50)
            print(f"  Critical runs: {cr['critical_count']:,} ({cr['critical_pct']}%)")
            for metric in ["defect_rate_pct", "downtime_mins", "efficiency_pct"]:
                if metric in cr:
                    m = cr[metric]
                    print(
                        f"  {metric}: critical={m['critical_mean']:.3f}  "
                        f"non-critical={m['non_critical_mean']:.3f}  "
                        f"delta={m['delta']:+.3f}"
                    )

        logger.info(f"Anomaly detection complete: {detector}")

        # ── PART 3: HTML REPORT ───────────────────────────────────
        logger.info("")
        logger.info("[HTML] Generating HTML report...")
        reporter = HTMLReporter(engine, detector)
        html_path = reporter.generate()
        print()
        print(f"  HTML report: {html_path}")

        # ── PART 4: CHARTS ────────────────────────────────────────
        try:
            from src.visualiser import Visualiser
            logger.info("")
            logger.info("[PLOT] Generating charts...")
            vis   = Visualiser(engine.df, detector.confirmed, engine.results)
            paths = vis.run_all()
            print()
            print("  CHARTS SAVED:")
            print("  " + "-" * 50)
            for p in paths:
                print(f"    {p}")
        except Exception as exc:
            logger.warning(f"[PLOT] Chart generation failed: {exc}")

        print()
        print("  NEXT: Open notebooks/manufacturing_deep_dive.ipynb")
        print("        for interactive visualisation and deeper exploration")


if __name__ == "__main__":
    main()
