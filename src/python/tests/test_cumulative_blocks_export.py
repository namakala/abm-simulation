"""Tests for export_results_assets — stage-7 figure and stats export.

Uses synthetic dataframes written to temp CSVs; no full simulation run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.python.demos.cumulative_blocks import (
    AGENT_STAT_COLUMNS,
    MODEL_STAT_COLUMNS,
    export_results_assets,
)
from src.python.visualization_utils import HAS_MATPLOTLIB

pytestmark = pytest.mark.unit


def _make_agent_df() -> pd.DataFrame:
    """Synthetic stage-7 agent data: 3 steps x 4 agents (12 agent-step rows)."""
    rows = []
    for step, base_pss, base_stress in [(1, 10.0, 0.10), (2, 20.0, 0.20), (3, 30.0, 0.30)]:
        for aid in range(4):
            rows.append(
                {
                    "Step": step,
                    "AgentID": aid + 1,
                    "pss10": base_pss + aid,
                    "resilience": 0.5,
                    "affect": 0.0,
                    "resources": 1.0,
                    "current_stress": base_stress + aid / 100,
                    "stress_controllability": 0.5,
                    "stress_overload": 0.1,
                    "consecutive_hindrances": float(step),
                }
            )
    return pd.DataFrame(rows)


def _make_model_df() -> pd.DataFrame:
    """Synthetic stage-7 model data: 3 daily rows."""
    return pd.DataFrame(
        {
            "avg_pss10": [15.0, 16.0, 17.0],
            "avg_stress": [0.4, 0.5, 0.6],
            "avg_affect": [-0.1, 0.0, 0.1],
            "avg_resilience": [0.5, 0.5, 0.5],
            "avg_resources": [0.6, 0.6, 0.6],
            "coping_success_rate": [0.4, 0.5, 0.6],
            "avg_challenge": [0.5, 0.5, 0.5],
            "avg_hindrance": [0.5, 0.5, 0.5],
            "avg_consecutive_hindrances": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def export_inputs(tmp_path):
    """Write synthetic stage-7 CSVs; return (model_csv, agent_csv, out_dir, figures_dir)."""
    model_csv = tmp_path / "stage7_model.csv"
    agent_csv = tmp_path / "stage7_agent.csv"
    _make_model_df().to_csv(model_csv, index=False)
    _make_agent_df().to_csv(agent_csv, index=False)
    return model_csv, agent_csv, tmp_path / "output", tmp_path / "figures"


def _read_stats(export_inputs) -> pd.DataFrame:
    model_csv, agent_csv, out, figs = export_inputs
    export_results_assets(model_csv, agent_csv, str(out), str(figs))
    return pd.read_csv(out / "stage7_results_stats.csv")


class TestExportResultsAssets:
    """export_results_assets renders figures and writes the stats CSV."""

    def test_returns_asset_paths(self, export_inputs):
        """Returns paths for all four assets, each existing on disk."""
        model_csv, agent_csv, out, figs = export_inputs
        assets = export_results_assets(model_csv, agent_csv, str(out), str(figs))
        assert set(assets) == {"initial_population", "final_population", "time_series", "stats_csv"}
        for path in assets.values():
            assert Path(path).exists()

    def test_writes_stats_csv_with_expected_schema(self, export_inputs):
        """Stats CSV is tidy long-format with the four documented levels."""
        df = _read_stats(export_inputs)
        assert list(df.columns) == ["level", "metric", "mean", "sd", "cv", "min", "max"]
        assert set(df["level"]) == {"population", "agent_step", "final_day", "correlation"}

    def test_population_stats_values(self, export_inputs):
        """Population rows summarize daily model metrics (mean, sd, cv, range)."""
        pop = _read_stats(export_inputs)
        pop = pop[pop["level"] == "population"].set_index("metric")
        assert pop.loc["avg_pss10", "mean"] == pytest.approx(16.0)
        assert pop.loc["avg_pss10", "sd"] == pytest.approx(1.0)
        assert pop.loc["avg_pss10", "cv"] == pytest.approx(6.25)
        assert pop.loc["avg_pss10", "min"] == pytest.approx(15.0)
        assert pop.loc["avg_pss10", "max"] == pytest.approx(17.0)
        assert set(pop.index) == set(MODEL_STAT_COLUMNS)

    def test_agent_step_stats_values(self, export_inputs):
        """Agent-step rows summarize all agent-step observations."""
        df = _read_stats(export_inputs)
        ag = df[df["level"] == "agent_step"].set_index("metric")
        pss = np.array([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33], dtype=float)
        assert ag.loc["pss10", "mean"] == pytest.approx(pss.mean())
        assert ag.loc["pss10", "sd"] == pytest.approx(pss.std(ddof=1))
        assert ag.loc["pss10", "cv"] == pytest.approx(pss.std(ddof=1) / pss.mean() * 100)
        assert set(ag.index) == set(AGENT_STAT_COLUMNS)

    def test_final_day_stats_values(self, export_inputs):
        """Final-day rows summarize the last-step agent slice only."""
        df = _read_stats(export_inputs)
        fd = df[df["level"] == "final_day"].set_index("metric")
        last = np.array([30, 31, 32, 33], dtype=float)
        assert fd.loc["pss10", "mean"] == pytest.approx(last.mean())
        assert fd.loc["pss10", "sd"] == pytest.approx(last.std(ddof=1))

    def test_correlations_initial_and_final(self, export_inputs):
        """Correlation rows hold initial and final pss10-stress Pearson r."""
        df = _read_stats(export_inputs)
        corr = df[df["level"] == "correlation"].set_index("metric")
        agent = _make_agent_df()
        initial = agent[agent["Step"] == 1]
        final = agent[agent["Step"] == 3]
        assert corr.loc["pss10_stress_initial", "mean"] == pytest.approx(
            np.corrcoef(initial["pss10"], initial["current_stress"])[0, 1]
        )
        assert corr.loc["pss10_stress_final", "mean"] == pytest.approx(
            np.corrcoef(final["pss10"], final["current_stress"])[0, 1]
        )
        assert math.isnan(corr.loc["pss10_stress_initial", "sd"])

    def test_zero_mean_cv_is_nan(self, export_inputs):
        """CV is NaN when the mean is zero (division by zero)."""
        pop = _read_stats(export_inputs)
        pop = pop[pop["level"] == "population"].set_index("metric")
        assert math.isnan(pop.loc["avg_affect", "cv"])

    def test_initial_csv_used_when_provided(self, tmp_path):
        """A pre-step initial CSV overrides the Step==1 slice for baseline assets."""
        model_csv = tmp_path / "m.csv"
        agent_csv = tmp_path / "a.csv"
        _make_model_df().to_csv(model_csv, index=False)
        _make_agent_df().to_csv(agent_csv, index=False)
        initial = pd.DataFrame(
            {
                "Step": [0, 0, 0, 0],
                "AgentID": [1, 2, 3, 4],
                "pss10": [10.0, 20.0, 30.0, 40.0],
                "resilience": [0.5] * 4,
                "affect": [0.0] * 4,
                "resources": [1.0] * 4,
                "current_stress": [0.40, 0.10, 0.30, 0.20],  # anti-correlated with pss10
                "stress_controllability": [0.5] * 4,
                "stress_overload": [0.1] * 4,
                "consecutive_hindrances": [0.0] * 4,
            }
        )
        initial_csv = tmp_path / "a_initial.csv"
        initial.to_csv(initial_csv, index=False)
        export_results_assets(model_csv, agent_csv, str(tmp_path / "out"), str(tmp_path / "figs"), initial_csv)
        df = pd.read_csv(tmp_path / "out" / "stage7_results_stats.csv")
        corr = df[df["level"] == "correlation"].set_index("metric")
        expected = np.corrcoef(initial["pss10"], initial["current_stress"])[0, 1]
        assert corr.loc["pss10_stress_initial", "mean"] == pytest.approx(expected)

    def test_minimal_columns_are_tolerated(self, tmp_path):
        """Missing columns are skipped, not fatal; correlations need both columns."""
        model_csv = tmp_path / "m.csv"
        agent_csv = tmp_path / "a.csv"
        _make_model_df().to_csv(model_csv, index=False)
        pd.DataFrame({"Step": [1, 1], "AgentID": [1, 2], "pss10": [10.0, 20.0]}).to_csv(agent_csv, index=False)
        assets = export_results_assets(model_csv, agent_csv, str(tmp_path / "out"), str(tmp_path / "figs"))
        df = pd.read_csv(tmp_path / "out" / "stage7_results_stats.csv")
        assert "correlation" not in set(df["level"])
        assert "pss10" in set(df[df["level"] == "agent_step"]["metric"])
        assert assets["initial_population"] == ""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib required for figure rendering")
    def test_figures_written_as_pdf(self, export_inputs):
        """All three figures are written as PDFs into figures_dir."""
        model_csv, agent_csv, out, figs = export_inputs
        export_results_assets(model_csv, agent_csv, str(out), str(figs))
        for name in (
            "full_model_initial_population.pdf",
            "full_model_final_population.pdf",
            "full_model_time_series.pdf",
        ):
            assert (figs / name).exists()
