# ================================================================
# src/html_reporter.py — HTMLReporter for PrecisionForge EDA
# ================================================================
# Generates a styled HTML report combining all EDA findings,
# statistical test results, feature importance, and anomaly summary.
# Opens directly in any browser without dependencies.
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from config import REPORTS_DIR, logger

OUTPUT_PATH = REPORTS_DIR / "manufacturing_eda_report.html"

_CSS = """
<style>
  body{font-family:system-ui,sans-serif;margin:0;padding:0;background:#f4f6f9;color:#1a202c}
  .header{background:#1a365d;color:#fff;padding:32px 40px}
  .header h1{margin:0;font-size:1.8rem}
  .header p{margin:6px 0 0;opacity:.8;font-size:.95rem}
  .container{max-width:1100px;margin:32px auto;padding:0 24px}
  .card{background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);
        margin-bottom:28px;overflow:hidden}
  .card-header{background:#2b6cb0;color:#fff;padding:14px 24px;font-weight:600;
               font-size:1rem}
  .card-body{padding:20px 24px}
  table{width:100%;border-collapse:collapse;font-size:.875rem}
  th{background:#ebf4ff;color:#2b6cb0;font-weight:600;padding:10px 14px;
     text-align:left;border-bottom:2px solid #bee3f8}
  td{padding:9px 14px;border-bottom:1px solid #e2e8f0}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:#f7fafc}
  .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.78rem;
         font-weight:600}
  .badge-sig{background:#c6f6d5;color:#276749}
  .badge-ns{background:#fed7d7;color:#9b2c2c}
  .badge-crit{background:#fed7d7;color:#9b2c2c}
  .badge-ok{background:#c6f6d5;color:#276749}
  .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px}
  .stat-box{background:#ebf4ff;border-radius:8px;padding:16px;text-align:center}
  .stat-box .val{font-size:1.6rem;font-weight:700;color:#2b6cb0}
  .stat-box .lbl{font-size:.8rem;color:#4a5568;margin-top:4px}
  .alert{background:#fff5f5;border-left:4px solid #fc8181;padding:12px 16px;
         border-radius:4px;margin-bottom:16px;font-size:.9rem}
  .note{background:#ebf8ff;border-left:4px solid #4299e1;padding:12px 16px;
        border-radius:4px;margin-bottom:16px;font-size:.9rem}
  .rank-bar{display:inline-block;height:10px;background:#4299e1;border-radius:3px;
            min-width:4px;vertical-align:middle;margin-right:8px}
</style>
"""


def _badge(sig: bool) -> str:
    cls  = "badge-sig" if sig else "badge-ns"
    text = "SIGNIFICANT" if sig else "not significant"
    return f'<span class="badge {cls}">{text}</span>'


def _trend_badge(trend: str) -> str:
    cls = "badge-crit" if trend == "worsening" else "badge-ok"
    return f'<span class="badge {cls}">{trend.upper()}</span>'


class HTMLReporter:
    """
    Generates a styled HTML report from EDAEngine and AnomalyDetector results.

    Usage:
        reporter = HTMLReporter(engine, detector)
        reporter.generate()   # saves to reports/manufacturing_eda_report.html
    """

    def __init__(self, engine, detector=None):
        self.engine   = engine
        self.detector = detector
        self.df       = engine.df

    # ── SECTION BUILDERS ─────────────────────────────────────────────

    def _overview(self) -> str:
        if "profile" not in self.engine.results:
            return ""
        p = self.engine.results["profile"]
        return f"""
        <div class="card">
          <div class="card-header">Dataset Overview</div>
          <div class="card-body">
            <div class="stat-grid">
              <div class="stat-box"><div class="val">{p['rows']:,}</div>
                <div class="lbl">Production Runs</div></div>
              <div class="stat-box"><div class="val">{p['columns']}</div>
                <div class="lbl">Columns</div></div>
              <div class="stat-box"><div class="val">{p['null_pct']}%</div>
                <div class="lbl">Missing Values</div></div>
              <div class="stat-box"><div class="val">{p['duplicates']:,}</div>
                <div class="lbl">Duplicate Rows</div></div>
              <div class="stat-box"><div class="val">{p['memory_mb']} MB</div>
                <div class="lbl">Memory Usage</div></div>
            </div>
          </div>
        </div>"""

    def _descriptive_stats(self) -> str:
        if "profile" not in self.engine.results:
            return ""
        p = self.engine.results["profile"]
        if "descriptive_stats" not in p:
            return ""
        stats = p["descriptive_stats"]
        mean_d   = stats.get("mean",  {})
        median_d = stats.get("50%",   {})
        std_d    = stats.get("std",   {})
        min_d    = stats.get("min",   {})
        max_d    = stats.get("max",   {})
        rows = ""
        for col in list(mean_d.keys())[:10]:
            rows += f"""
            <tr>
              <td><strong>{col}</strong></td>
              <td>{mean_d.get(col, ''):.3f}</td>
              <td>{median_d.get(col, ''):.3f}</td>
              <td>{std_d.get(col, ''):.3f}</td>
              <td>{min_d.get(col, ''):.3f}</td>
              <td>{max_d.get(col, ''):.3f}</td>
            </tr>"""
        return f"""
        <div class="card">
          <div class="card-header">Descriptive Statistics</div>
          <div class="card-body">
            <table>
              <tr><th>Column</th><th>Mean</th><th>Median</th>
                  <th>Std Dev</th><th>Min</th><th>Max</th></tr>
              {rows}
            </table>
          </div>
        </div>"""

    def _group_analysis(self) -> str:
        if "group_analysis" not in self.engine.results:
            return ""
        html = ""
        for cat_col, metrics in self.engine.results["group_analysis"].items():
            for num_col, rows in metrics.items():
                row_html = ""
                for row in rows:
                    g = str(row.get(cat_col, ""))
                    row_html += f"""
                    <tr>
                      <td>{row.get('rank', '')}</td>
                      <td><strong>{g}</strong></td>
                      <td>{row.get('mean', 0):.3f}</td>
                      <td>{row.get('median', 0):.3f}</td>
                      <td>{row.get('std', 0):.3f}</td>
                      <td>{row.get('count', 0):,}</td>
                    </tr>"""
                html += f"""
                <div class="card">
                  <div class="card-header">
                    {num_col.replace('_', ' ').title()} by
                    {cat_col.replace('_', ' ').title()}
                  </div>
                  <div class="card-body">
                    <table>
                      <tr><th>Rank</th><th>Group</th><th>Mean</th>
                          <th>Median</th><th>Std Dev</th><th>Count</th></tr>
                      {row_html}
                    </table>
                  </div>
                </div>"""
        return html

    def _statistical_tests(self) -> str:
        html = ""
        for test_key, test_name, stat_label in [
            ("anova",   "ANOVA Tests",            "F"),
            ("kruskal", "Kruskal-Wallis Tests",   "H"),
        ]:
            if test_key not in self.engine.results:
                continue
            rows = ""
            for key, res in self.engine.results[test_key].items():
                stat_val = res.get("f_stat") or res.get("h_stat", 0)
                rows += f"""
                <tr>
                  <td>{key}</td>
                  <td>{stat_val:.3f}</td>
                  <td>{res['p_value']:.6f}</td>
                  <td>{_badge(res['significant'])}</td>
                </tr>"""
            html += f"""
            <div class="card">
              <div class="card-header">{test_name}</div>
              <div class="card-body">
                <div class="note">
                  p &lt; 0.05 → group means differ significantly.
                  Kruskal-Wallis is the non-parametric confirmation.
                </div>
                <table>
                  <tr><th>Test</th><th>{stat_label}-stat</th>
                      <th>p-value</th><th>Result</th></tr>
                  {rows}
                </table>
              </div>
            </div>"""
        return html

    def _feature_importance(self) -> str:
        if "feature_importance" not in self.engine.results:
            return ""
        feats = self.engine.results["feature_importance"]
        max_r = feats[0]["abs_r"] if feats else 1
        rows  = ""
        for i, f in enumerate(feats, 1):
            bar_w = int(f["abs_r"] / max(max_r, 0.001) * 120)
            rows += f"""
            <tr>
              <td>{i}</td>
              <td><strong>{f['feature']}</strong></td>
              <td>
                <span class="rank-bar" style="width:{bar_w}px"></span>
                {f['abs_r']:.4f}
              </td>
              <td>{f['r']:+.4f}</td>
              <td>{f['direction']}</td>
              <td>{'✓' if f['significant'] else '—'}</td>
            </tr>"""
        return f"""
        <div class="card">
          <div class="card-header">Feature Importance — Pearson |r| with defect_rate_pct</div>
          <div class="card-body">
            <div class="note">Higher |r| = stronger linear predictor of defect rate.</div>
            <table>
              <tr><th>#</th><th>Feature</th><th>|r|</th>
                  <th>r (signed)</th><th>Direction</th><th>p &lt; 0.05</th></tr>
              {rows}
            </table>
          </div>
        </div>"""

    def _forecast(self) -> str:
        if "forecast" not in self.engine.results:
            return ""
        html = '<div class="card"><div class="card-header">Forecast — Exponential Smoothing</div><div class="card-body">'
        for col, res in self.engine.results["forecast"].items():
            html += f"""
            <div class="stat-grid" style="margin-bottom:16px">
              <div class="stat-box">
                <div class="val">{res['last_actual']:.3f}</div>
                <div class="lbl">Last Period Actual</div>
              </div>
              <div class="stat-box">
                <div class="val">{res['next_period']:.3f}</div>
                <div class="lbl">Next Period Forecast</div>
              </div>
              <div class="stat-box">
                <div class="val">{_trend_badge(res['trend'])}</div>
                <div class="lbl">Trend</div>
              </div>
              <div class="stat-box">
                <div class="val">{res['periods_used']}</div>
                <div class="lbl">Periods Used</div>
              </div>
            </div>"""
            if res["trend"] == "worsening":
                html += '<div class="alert">⚠ Forecast indicates a worsening trend. Escalate to operations leadership.</div>'
        html += "</div></div>"
        return html

    def _critical_runs(self) -> str:
        if self.detector is None:
            return ""
        if "critical_runs" not in self.detector.results:
            return ""
        cr    = self.detector.results["critical_runs"]
        rows  = ""
        skip  = {"critical_count", "non_critical_count", "critical_pct"}
        for metric, vals in cr.items():
            if metric in skip or not isinstance(vals, dict):
                continue
            delta_class = "badge-crit" if vals['delta'] > 0 else "badge-ok"
            rows += f"""
            <tr>
              <td><strong>{metric}</strong></td>
              <td>{vals['critical_mean']:.3f}</td>
              <td>{vals['non_critical_mean']:.3f}</td>
              <td><span class="badge {delta_class}">{vals['delta']:+.3f}</span></td>
              <td>{vals['ratio']:.2f}×</td>
            </tr>"""
        return f"""
        <div class="card">
          <div class="card-header">Critical-Severity Run Profile</div>
          <div class="card-body">
            <div class="stat-grid" style="margin-bottom:20px">
              <div class="stat-box">
                <div class="val">{cr['critical_count']:,}</div>
                <div class="lbl">Critical Runs</div>
              </div>
              <div class="stat-box">
                <div class="val">{cr['critical_pct']}%</div>
                <div class="lbl">of All Runs</div>
              </div>
              <div class="stat-box">
                <div class="val">{cr['non_critical_count']:,}</div>
                <div class="lbl">Non-Critical Runs</div>
              </div>
            </div>
            <table>
              <tr><th>Metric</th><th>Critical Mean</th><th>Non-Critical Mean</th>
                  <th>Delta</th><th>Ratio</th></tr>
              {rows}
            </table>
          </div>
        </div>"""

    def _anomaly_summary(self) -> str:
        if self.detector is None or not self.detector.results:
            return ""
        rows = ""
        for col, vals in self.detector.results.items():
            if col == "critical_runs" or not isinstance(vals, dict):
                continue
            rows += f"""
            <tr>
              <td><strong>{col}</strong></td>
              <td>{vals.get('iqr_flagged', 0):,}</td>
              <td>{vals.get('zscore_flagged', 0):,}</td>
              <td>{vals.get('confirmed_anomalies', 0):,}</td>
              <td>{vals.get('anomaly_pct', 0):.2f}%</td>
            </tr>"""
        if not rows:
            return ""
        n_confirmed = len(self.detector.confirmed) if self.detector.confirmed is not None else 0
        return f"""
        <div class="card">
          <div class="card-header">Anomaly Detection Summary (IQR + Z-score consensus)</div>
          <div class="card-body">
            <p style="margin:0 0 14px">
              <strong>{n_confirmed:,} confirmed anomaly rows</strong> saved to
              <code>reports/anomalies.csv</code>
            </p>
            <table>
              <tr><th>Column</th><th>IQR Flagged</th><th>Z-score Flagged</th>
                  <th>Confirmed</th><th>Rate</th></tr>
              {rows}
            </table>
          </div>
        </div>"""

    # ── PUBLIC INTERFACE ──────────────────────────────────────────────

    def generate(self, path: pathlib.Path = None) -> pathlib.Path:
        """Build the full HTML report and save it."""
        out = path or OUTPUT_PATH
        body = (
            self._overview()
            + self._descriptive_stats()
            + self._group_analysis()
            + self._statistical_tests()
            + self._feature_importance()
            + self._forecast()
            + self._critical_runs()
            + self._anomaly_summary()
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PrecisionForge — Manufacturing EDA Report</title>
  {_CSS}
</head>
<body>
  <div class="header">
    <h1>PrecisionForge Manufacturing Group — EDA Report</h1>
    <p>Production data analysis · Defect rate investigation · Board-ready findings</p>
  </div>
  <div class="container">
    {body}
  </div>
</body>
</html>"""

        out.write_text(html, encoding="utf-8")
        logger.info(f"[HTML] Report saved: {out}")
        return out
