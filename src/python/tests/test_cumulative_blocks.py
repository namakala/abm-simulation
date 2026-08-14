"""Tests for cumulative_blocks.py — 7-stage cumulative block demo runner."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.python.demos.cumulative_blocks import (
    METRIC_COLUMNS,
    STAGES,
    compute_stage7_row,
    compute_stage_row,
    run_cumulative_demo,
    run_stage,
    run_stage7,
)


class TestStageRegistry:
    """Stage registry defines a cumulative block build-up."""

    def test_has_seven_stages(self):
        """STAGES contains exactly 7 stages ending with the full model."""
        assert len(STAGES) == 7
        assert STAGES[-1]["id"] == 7
        assert STAGES[-1]["name"] == "full_model"

    def test_blocks_accumulate(self):
        """Each stage's blocks are a superset of the previous stage's."""
        for prev, cur in zip(STAGES, STAGES[1:]):
            assert set(prev["blocks_active"]) <= set(cur["blocks_active"])

    def test_core_block_names(self):
        """The six progressive blocks use expected names."""
        assert STAGES[0]["blocks_active"] == ["initialization"]
        assert "stress_perception" in STAGES[1]["blocks_active"]
        assert "resilience_activation" in STAGES[2]["blocks_active"]
        assert "interaction" in STAGES[3]["blocks_active"]
        assert "resource_allocation" in STAGES[4]["blocks_active"]
        assert "stress_buffering" in STAGES[5]["blocks_active"]


class TestRunStage:
    """Staged step loop over a real StressModel."""

    def test_deterministic(self):
        """Same seed produces identical daily summaries."""
        run1 = run_stage(2, agents=8, days=3, seed=42)
        run2 = run_stage(2, agents=8, days=3, seed=42)
        assert run1.daily_summaries == run2.daily_summaries

    def test_collects_metrics_for_active_blocks(self):
        """Stage 4 collects perception, activation, and interaction metrics."""
        run = run_stage(4, agents=8, days=3, seed=42)
        assert len(run.metrics["perception"]) > 0
        assert len(run.metrics["activation"]) > 0
        assert len(run.metrics["interaction"]) > 0
        # Daily blocks not yet active
        assert run.metrics["allocation"] == []
        assert run.metrics["buffering"] == []

    def test_stage1_has_no_dynamics(self):
        """Stage 1 collects initialization metrics without stepping."""
        run = run_stage(1, agents=8, days=3, seed=42)
        assert len(run.metrics["initialization"]) == 8
        assert len(run.daily_summaries) == 1


class TestComputeStageRow:
    """Stage rows for cumulative_stages_metrics.csv."""

    def test_columns_present(self):
        """Row contains all METRIC_COLUMNS plus stage and blocks_active."""
        run = run_stage(4, agents=8, days=3, seed=42)
        row = compute_stage_row(STAGES[3], 8, 3, run)
        assert row["stage"] == 4
        assert "stress_perception" in row["blocks_active"]
        for col in METRIC_COLUMNS:
            assert col in row

    def test_inactive_blocks_are_nan(self):
        """Buffering metrics are NaN for stage 4 (buffering not yet active)."""
        run = run_stage(4, agents=8, days=3, seed=42)
        row = compute_stage_row(STAGES[3], 8, 3, run)
        assert math.isnan(row["avg_buffering_strength"])
        assert math.isnan(row["avg_stress_depletion"])

    def test_active_blocks_populated(self):
        """Stage 6 populates buffering and allocation metrics."""
        run = run_stage(6, agents=8, days=3, seed=42)
        row = compute_stage_row(STAGES[5], 8, 3, run)
        assert not math.isnan(row["avg_buffering_strength"])
        assert not math.isnan(row["avg_regeneration"])


class TestRunStage7:
    """Full-model stage invokes simulate.py."""

    def test_invokes_simulate(self, tmp_path, monkeypatch):
        """run_stage7 calls simulate.py with expected arguments."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))

        monkeypatch.setattr("src.python.demos.cumulative_blocks.subprocess.run", fake_run)
        model_csv, agent_csv = run_stage7(tmp_path, agents=100, days=90, seed=42)

        cmd, kwargs = calls[0]
        assert "simulate.py" in cmd
        assert "--days" in cmd and "90" in cmd
        assert "--seed" in cmd and "42" in cmd
        assert "--agents" in cmd and "100" in cmd
        assert "--prefix" in cmd and "stage7" in cmd
        assert "--output-data" in cmd
        assert kwargs["cwd"].joinpath("simulate.py").exists()
        assert kwargs["check"] is True
        assert model_csv.name == "stage7_model.csv"
        assert agent_csv.name == "stage7_agent.csv"


class TestComputeStage7Row:
    """Stage 7 row from simulate.py CSVs."""

    def _write_fake_outputs(self, tmp_path):
        model_df = pd.DataFrame(
            {
                "day": [0, 1, 2],
                "avg_pss10": [20.0, 21.0, 22.0],
                "avg_stress": [0.4, 0.41, 0.42],
                "avg_affect": [0.0, -0.01, -0.02],
                "avg_resilience": [0.5, 0.51, 0.52],
                "avg_resources": [0.6, 0.61, 0.62],
                "stress_prevalence": [0.3, 0.31, 0.32],
                "avg_challenge": [0.5, 0.51, 0.52],
                "avg_hindrance": [0.5, 0.49, 0.48],
                "coping_success_rate": [0.4, 0.41, 0.42],
                "stress_events": [400.0, 410.0, 420.0],
                "social_interactions": [300.0, 310.0, 320.0],
                "daily_social_support_rate": [0.3, 0.31, 0.32],
            }
        )
        model_csv = tmp_path / "stage7_model.csv"
        model_df.to_csv(model_csv, index=False)
        (tmp_path / "stage7_agent.csv").write_text("AgentID,Step,pss10\n")
        return model_csv

    def test_state_columns_from_last_row(self, tmp_path):
        """Stage 7 row reads final-day state from the model CSV."""
        model_csv = self._write_fake_outputs(tmp_path)
        row = compute_stage7_row(model_csv, agents=100, days=3, stage=STAGES[6])
        assert row["stage"] == 7
        assert row["avg_pss10"] == pytest.approx(22.0)
        assert row["avg_stress"] == pytest.approx(0.42)
        assert row["stress_prevalence"] == pytest.approx(0.32)

    def test_phase_columns_from_run_means(self, tmp_path):
        """Stage 7 row averages phase metrics over the run."""
        model_csv = self._write_fake_outputs(tmp_path)
        row = compute_stage7_row(model_csv, agents=100, days=3, stage=STAGES[6])
        assert row["avg_challenge"] == pytest.approx(0.51)
        assert row["coping_success_rate"] == pytest.approx(0.41)
        assert row["stress_event_rate"] == pytest.approx(4.1)
        assert row["interactions_per_agent_day"] == pytest.approx(3.1)
        # Not present in simulate.py output
        assert math.isnan(row["avg_regeneration"])
        assert math.isnan(row["avg_buffering_strength"])


class TestRunCumulativeDemo:
    """End-to-end runner writes all expected outputs."""

    def test_writes_all_outputs(self, tmp_path, monkeypatch):
        """run_cumulative_demo writes metrics CSV, per-stage summaries, JSON, and figures."""
        out = tmp_path
        # Fake stage 7 outputs (subprocess is mocked)
        pd.DataFrame(
            {
                "day": [0, 1],
                "avg_pss10": [20.0, 21.0],
                "avg_stress": [0.4, 0.41],
                "avg_affect": [0.0, 0.0],
                "avg_resilience": [0.5, 0.51],
                "avg_resources": [0.6, 0.61],
                "stress_prevalence": [0.3, 0.31],
                "avg_challenge": [0.5, 0.51],
                "avg_hindrance": [0.5, 0.49],
                "coping_success_rate": [0.4, 0.41],
                "stress_events": [40.0, 41.0],
                "social_interactions": [30.0, 31.0],
                "daily_social_support_rate": [0.3, 0.31],
            }
        ).to_csv(out / "stage7_model.csv", index=False)
        pd.DataFrame(
            {
                "Step": [1, 1, 1, 2, 2, 2],
                "AgentID": [1, 2, 3, 1, 2, 3],
                "pss10": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
                "resilience": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "affect": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "resources": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "current_stress": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
                "stress_controllability": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "stress_overload": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "consecutive_hindrances": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            }
        ).to_csv(out / "stage7_agent.csv", index=False)

        monkeypatch.setattr(
            "src.python.demos.cumulative_blocks.subprocess.run",
            lambda *a, **k: None,
        )

        figs = tmp_path / "figures"
        result = run_cumulative_demo(agents=8, days=3, seed=42, output_dir=str(out), figures_dir=str(figs))

        assert result == out
        metrics_csv = out / "cumulative_stages_metrics.csv"
        assert metrics_csv.exists()
        df = pd.read_csv(metrics_csv)
        assert len(df) == 7
        for col in METRIC_COLUMNS:
            assert col in df.columns
        for n in range(1, 7):
            assert (out / f"cumulative_stage{n}_summary.csv").exists()
        meta = json.loads((out / "cumulative_blocks.json").read_text())
        assert meta["seed"] == 42
        assert len(meta["stages"]) == 7
        # Stage-7 article assets
        assert (out / "stage7_results_stats.csv").exists()
        for name in (
            "full_model_initial_population.pdf",
            "full_model_final_population.pdf",
            "full_model_time_series.pdf",
        ):
            assert (figs / name).exists()
