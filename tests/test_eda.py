# ================================================================
# tests/test_eda.py — Unit Tests for PrecisionForge Module 06
# ================================================================
# HOW TO RUN:
#   python tests/test_eda.py
#   pytest tests/
# ================================================================

import sys, pathlib
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from src.eda_engine       import EDAEngine
from src.anomaly_detector import AnomalyDetector


# ── TEST DATA FACTORY ─────────────────────────────────────────────

def make_sample_df(rows: int = 60) -> pd.DataFrame:
    """Small, predictable manufacturing DataFrame for unit tests."""
    np.random.seed(42)
    plant_types = ["Automotive", "Pharmaceutical", "Precision", "Food", "Steel", "Advanced"]
    shifts      = ["Morning", "Afternoon", "Night"]
    eq_types    = ["CNC Machine", "Conveyor Belt", "Welding Robot", "Packaging Machine"]
    eq_status   = ["Operational", "Maintenance"]
    defect_types = ["Assembly Error", "Dimensional", "Electrical Fault",
                    "Material Defect", "Surface Finish", "None"]
    severities  = ["Critical", "Major", "Minor", "None"]

    defect_rate = np.clip(np.random.normal(2.5, 2.0, rows), 0, 10).round(3)

    return pd.DataFrame({
        "run_id":                range(1, rows + 1),
        "run_date":              ["2024-06-15"] * rows,
        "shift":                 [shifts[i % len(shifts)] for i in range(rows)],
        "plant_type":            [plant_types[i % len(plant_types)] for i in range(rows)],
        "planned_units":         np.random.randint(500, 2000, rows),
        "actual_units":          np.random.randint(400, 1900, rows),
        "defective_units":       np.random.randint(0, 100, rows),
        "efficiency_pct":        np.random.uniform(60, 100, rows).round(2),
        "downtime_mins":         np.random.uniform(0, 180, rows).round(1),
        "yield_pct":             np.random.uniform(70, 100, rows).round(2),
        "defect_rate_pct":       defect_rate,
        "primary_defect_type":   [defect_types[i % len(defect_types)] for i in range(rows)],
        "primary_severity":      [severities[i % len(severities)] for i in range(rows)],
        "equipment_type":        [eq_types[i % len(eq_types)] for i in range(rows)],
        "equipment_status":      [eq_status[i % len(eq_status)] for i in range(rows)],
        "equipment_efficiency_pct": np.random.uniform(70, 100, rows).round(2),
        "days_since_maintenance":   np.random.uniform(0, 90, rows).round(1),
    })


def _setup_engine(rows: int = 60) -> EDAEngine:
    """Return an engine with df/num_cols/cat_cols pre-loaded."""
    engine = EDAEngine()
    engine.df = make_sample_df(rows)
    id_like   = {"run_id"}
    engine.num_cols = [
        c for c in engine.df.select_dtypes(include=["number"]).columns
        if c not in id_like
    ]
    engine.cat_cols = engine.df.select_dtypes(include=["object"]).columns.tolist()
    engine._status  = "loaded"
    return engine


# ── PROFILE TESTS ─────────────────────────────────────────────────

def test_profile_row_count():
    engine = _setup_engine(60)
    engine.profile()
    assert engine.results["profile"]["rows"] == 60
    print("  PASS: test_profile_row_count")


def test_profile_null_count_clean_data():
    engine = _setup_engine()
    engine.profile()
    assert engine.results["profile"]["total_nulls"] == 0
    print("  PASS: test_profile_null_count_clean_data")


def test_profile_stores_descriptive_stats():
    engine = _setup_engine()
    engine.profile()
    assert "descriptive_stats" in engine.results["profile"]
    assert "mean" in engine.results["profile"]["descriptive_stats"]
    print("  PASS: test_profile_stores_descriptive_stats")


def test_profile_memory_is_positive():
    engine = _setup_engine()
    engine.profile()
    assert engine.results["profile"]["memory_mb"] > 0
    print("  PASS: test_profile_memory_is_positive")


# ── GROUP ANALYSIS TESTS ──────────────────────────────────────────

def test_group_analysis_produces_results():
    engine = _setup_engine()
    engine.group_analysis()
    assert "group_analysis" in engine.results
    assert len(engine.results["group_analysis"]) > 0
    print("  PASS: test_group_analysis_produces_results")


def test_group_analysis_rank_starts_at_one():
    engine = _setup_engine()
    engine.group_analysis()
    first_cat = list(engine.results["group_analysis"].keys())[0]
    first_num = list(engine.results["group_analysis"][first_cat].keys())[0]
    top_row   = engine.results["group_analysis"][first_cat][first_num][0]
    assert top_row["rank"] == 1
    print("  PASS: test_group_analysis_rank_starts_at_one")


def test_group_analysis_custom_columns():
    engine = _setup_engine()
    engine.group_analysis(group_col="shift", value_col="defect_rate_pct")
    assert "shift" in engine.results["group_analysis"]
    print("  PASS: test_group_analysis_custom_columns")


# ── CORRELATION TESTS ─────────────────────────────────────────────

def test_correlation_only_strong_pairs():
    from config import CORRELATION_THRESHOLD
    engine = _setup_engine()
    engine.correlation()
    if "correlation" in engine.results:
        for pair in engine.results["correlation"]["strong_pairs"]:
            assert abs(pair["correlation"]) >= CORRELATION_THRESHOLD
    print("  PASS: test_correlation_only_strong_pairs")


def test_method_chaining_returns_self():
    engine = _setup_engine()
    result = engine.profile().group_analysis().correlation()
    assert result is engine
    print("  PASS: test_method_chaining_returns_self")


# ── ANOVA / KRUSKAL TESTS ─────────────────────────────────────────

def test_anova_produces_results():
    engine = _setup_engine()
    engine.anova_test()
    assert "anova" in engine.results
    assert len(engine.results["anova"]) > 0
    print("  PASS: test_anova_produces_results")


def test_anova_result_has_f_and_p():
    engine = _setup_engine()
    engine.anova_test()
    for key, res in engine.results["anova"].items():
        assert "f_stat"  in res
        assert "p_value" in res
        assert "significant" in res
    print("  PASS: test_anova_result_has_f_and_p")


def test_kruskal_produces_results():
    engine = _setup_engine()
    engine.kruskal_test()
    assert "kruskal" in engine.results
    assert len(engine.results["kruskal"]) > 0
    print("  PASS: test_kruskal_produces_results")


def test_kruskal_result_has_h_and_p():
    engine = _setup_engine()
    engine.kruskal_test()
    for key, res in engine.results["kruskal"].items():
        assert "h_stat"  in res
        assert "p_value" in res
    print("  PASS: test_kruskal_result_has_h_and_p")


# ── FEATURE IMPORTANCE TESTS ──────────────────────────────────────

def test_feature_importance_produces_rankings():
    engine = _setup_engine()
    engine.feature_importance()
    assert "feature_importance" in engine.results
    assert len(engine.results["feature_importance"]) > 0
    print("  PASS: test_feature_importance_produces_rankings")


def test_feature_importance_sorted_by_abs_r():
    engine = _setup_engine()
    engine.feature_importance()
    rankings = engine.results["feature_importance"]
    for i in range(len(rankings) - 1):
        assert rankings[i]["abs_r"] >= rankings[i + 1]["abs_r"]
    print("  PASS: test_feature_importance_sorted_by_abs_r")


def test_feature_importance_excludes_target():
    engine = _setup_engine()
    engine.feature_importance(target_col="defect_rate_pct")
    features = [f["feature"] for f in engine.results["feature_importance"]]
    assert "defect_rate_pct" not in features
    print("  PASS: test_feature_importance_excludes_target")


# ── ANOMALY DETECTOR TESTS ────────────────────────────────────────

def test_anomaly_detector_flags_obvious_outlier():
    df = make_sample_df(30)
    df.loc[0, "defect_rate_pct"] = 999.0   # extreme outlier
    detector = AnomalyDetector(df)
    detector.run(columns=["defect_rate_pct"])
    assert 0 in detector.confirmed.index
    print("  PASS: test_anomaly_detector_flags_obvious_outlier")


def test_anomaly_detector_clean_data_few_anomalies():
    np.random.seed(42)
    df = pd.DataFrame({"value": np.random.normal(50, 5, 200)})
    detector = AnomalyDetector(df)
    detector.run(columns=["value"])
    anomaly_pct = len(detector.confirmed) / len(df) * 100
    assert anomaly_pct < 2.0
    print("  PASS: test_anomaly_detector_clean_data_few_anomalies")


def test_anomaly_detector_summary_columns():
    df = make_sample_df()
    detector = AnomalyDetector(df)
    detector.run()
    summary = detector.summary()
    for col in ["column", "iqr_flagged", "zscore_flagged", "confirmed_anomalies", "anomaly_pct"]:
        assert col in summary.columns
    print("  PASS: test_anomaly_detector_summary_columns")


def test_anomaly_detector_does_not_modify_original():
    df = make_sample_df()
    original_val = df.loc[0, "defect_rate_pct"]
    AnomalyDetector(df).run()
    assert df.loc[0, "defect_rate_pct"] == original_val
    print("  PASS: test_anomaly_detector_does_not_modify_original")


def test_zero_std_column_no_crash():
    df = pd.DataFrame({
        "constant":       [5.0] * 30,
        "defect_rate_pct": np.random.uniform(0, 10, 30),
    })
    try:
        AnomalyDetector(df).run(columns=["constant", "defect_rate_pct"])
        print("  PASS: test_zero_std_column_no_crash")
    except Exception as e:
        raise AssertionError(f"Crashed on zero-std column: {e}")


# ── FLAG CRITICAL RUNS TESTS ──────────────────────────────────────

def test_flag_critical_runs_produces_profile():
    df = make_sample_df()
    detector = AnomalyDetector(df)
    detector.flag_critical_runs()
    assert "critical_runs" in detector.results
    cr = detector.results["critical_runs"]
    assert "critical_count" in cr
    assert "critical_pct"   in cr
    print("  PASS: test_flag_critical_runs_produces_profile")


def test_flag_critical_runs_metrics_present():
    df = make_sample_df()
    detector = AnomalyDetector(df)
    detector.flag_critical_runs()
    cr = detector.results["critical_runs"]
    assert "defect_rate_pct" in cr
    m = cr["defect_rate_pct"]
    assert "critical_mean"     in m
    assert "non_critical_mean" in m
    assert "delta"             in m
    print("  PASS: test_flag_critical_runs_metrics_present")


# ── TEST RUNNER ───────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  PRECISIONFORGE — EDA UNIT TESTS (MODULE 06)")
    print("=" * 60)

    print("\n  Profile tests:")
    test_profile_row_count()
    test_profile_null_count_clean_data()
    test_profile_stores_descriptive_stats()
    test_profile_memory_is_positive()

    print("\n  Group analysis tests:")
    test_group_analysis_produces_results()
    test_group_analysis_rank_starts_at_one()
    test_group_analysis_custom_columns()

    print("\n  Correlation tests:")
    test_correlation_only_strong_pairs()
    test_method_chaining_returns_self()

    print("\n  ANOVA / Kruskal-Wallis tests:")
    test_anova_produces_results()
    test_anova_result_has_f_and_p()
    test_kruskal_produces_results()
    test_kruskal_result_has_h_and_p()

    print("\n  Feature importance tests:")
    test_feature_importance_produces_rankings()
    test_feature_importance_sorted_by_abs_r()
    test_feature_importance_excludes_target()

    print("\n  AnomalyDetector tests:")
    test_anomaly_detector_flags_obvious_outlier()
    test_anomaly_detector_clean_data_few_anomalies()
    test_anomaly_detector_summary_columns()
    test_anomaly_detector_does_not_modify_original()
    test_zero_std_column_no_crash()

    print("\n  flag_critical_runs tests:")
    test_flag_critical_runs_produces_profile()
    test_flag_critical_runs_metrics_present()

    print()
    print("=" * 60)
    print("  All tests passed ✓")
    print("=" * 60)
