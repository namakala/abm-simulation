"""7-stage cumulative building-block demo runner.

Stages 1-6 run a staged step loop over a real ``StressModel`` re-
instantiated per stage with the same seed (identical population and
network across stages, so block effects are comparable). Stage 7 runs
the production ``simulate.py`` entry point. All results are exported as
CSV/JSON to ``data/output`` for the article's results section.

Block order mirrors ``Person.step()``: subevent loop (perception ->
activation, interaction) then daily consolidation (allocation,
buffering). The remaining daily-cycle internals (affect dynamics,
PSS-10, daily reset) and network adaptation appear only in stage 7.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.python.agent import get_neighbor_affects
from src.python.config import get_config
from src.python.demos.module_metrics import (
    aggregate_activation,
    aggregate_allocation,
    aggregate_buffering,
    aggregate_interaction,
    aggregate_perception,
    extract_activation_metrics,
    extract_allocation_metrics,
    extract_buffering_metrics,
    extract_initialization_metrics,
    extract_interaction_metrics,
    extract_perception_metrics,
)
from src.python.math_utils import sample_poisson
from src.python.model import StressModel
from src.python.phases import (
    run_resilience_activation,
    run_resource_allocation,
    run_stress_buffering,
    run_stress_perception,
)
from src.python.phases.interaction import process_interaction as phase_process_interaction
from src.python.visualization_utils import (
    create_time_series_visualization,
    create_visualization_report,
)

DEFAULT_AGENTS = 100
DEFAULT_DAYS = 90
DEFAULT_SEED = 42

STAGES: List[Dict[str, Any]] = [
    {"id": 1, "name": "initialization", "blocks_active": ["initialization"]},
    {"id": 2, "name": "stress_perception", "blocks_active": ["initialization", "stress_perception"]},
    {
        "id": 3,
        "name": "resilience_activation",
        "blocks_active": ["initialization", "stress_perception", "resilience_activation"],
    },
    {
        "id": 4,
        "name": "interaction",
        "blocks_active": ["initialization", "stress_perception", "resilience_activation", "interaction"],
    },
    {
        "id": 5,
        "name": "resource_allocation",
        "blocks_active": [
            "initialization",
            "stress_perception",
            "resilience_activation",
            "interaction",
            "resource_allocation",
        ],
    },
    {
        "id": 6,
        "name": "stress_buffering",
        "blocks_active": [
            "initialization",
            "stress_perception",
            "resilience_activation",
            "interaction",
            "resource_allocation",
            "stress_buffering",
        ],
    },
    {
        "id": 7,
        "name": "full_model",
        "blocks_active": [
            "initialization",
            "stress_perception",
            "resilience_activation",
            "interaction",
            "resource_allocation",
            "stress_buffering",
            "affect_dynamics",
            "pss10_consolidation",
            "daily_reset",
            "network_adaptation",
        ],
    },
]

METRIC_COLUMNS = [
    "avg_pss10",
    "avg_stress",
    "avg_affect",
    "avg_resilience",
    "avg_resources",
    "stress_prevalence",
    "avg_challenge",
    "avg_hindrance",
    "stress_event_rate",
    "coping_success_rate",
    "interactions_per_agent_day",
    "support_exchange_rate",
    "avg_regeneration",
    "avg_net_resource_change",
    "avg_buffering_strength",
    "avg_stress_depletion",
]

# Stage-7 export columns summarized into stage7_results_stats.csv.
MODEL_STAT_COLUMNS = [
    "avg_pss10",
    "avg_stress",
    "avg_affect",
    "avg_resilience",
    "avg_resources",
    "coping_success_rate",
    "avg_challenge",
    "avg_hindrance",
    "avg_consecutive_hindrances",
]
AGENT_STAT_COLUMNS = [
    "pss10",
    "resilience",
    "affect",
    "resources",
    "current_stress",
    "stress_controllability",
    "stress_overload",
    "consecutive_hindrances",
]

# Figure filenames written by export_results_assets.
FIGURE_FILENAMES = {
    "initial_population": "full_model_initial_population.pdf",
    "final_population": "full_model_final_population.pdf",
    "time_series": "full_model_time_series.pdf",
}

_METRIC_BUCKETS = ("initialization", "perception", "activation", "interaction", "allocation", "buffering")


class StageRun:
    """Outcome of a staged run for one stage.

    Attributes:
        metrics: Per-block metric buckets (lists of module_metrics dicts).
        daily_summaries: Per-day population means over the staged run.
        model: The final StressModel (post-run agent snapshots).
    """

    def __init__(
        self,
        metrics: Dict[str, List[Dict[str, Any]]],
        daily_summaries: List[Dict[str, float]],
        model: StressModel,
    ) -> None:
        self.metrics = metrics
        self.daily_summaries = daily_summaries
        self.model = model


def _empty_metric_buckets() -> Dict[str, List[Dict[str, Any]]]:
    """Create an empty metric bucket per block family."""
    return {key: [] for key in _METRIC_BUCKETS}


def _population_summary(model: StressModel, day: int) -> Dict[str, float]:
    """Compute one day's population means from final agent attributes."""
    agents = list(model.agents)
    return {
        "day": float(day),
        "avg_pss10": float(np.mean([a.pss10 for a in agents])),
        "avg_stress": float(np.mean([a.current_stress for a in agents])),
        "avg_affect": float(np.mean([a.affect for a in agents])),
        "avg_resilience": float(np.mean([a.resilience for a in agents])),
        "avg_resources": float(np.mean([a.resources for a in agents])),
        "stress_prevalence": float(np.mean([1.0 if a.stressed else 0.0 for a in agents])),
    }


def _apply_interaction_delta(state: Dict[str, Any], delta: Dict[str, Any]) -> None:
    """Apply an interaction delta additively (matching Person.step)."""
    for key, value in delta.items():
        if key in state and isinstance(state[key], dict) and isinstance(value, dict):
            state[key].update(value)
        elif key in state and isinstance(state[key], (int, float)) and isinstance(value, (int, float)):
            state[key] = state[key] + value
        else:
            state[key] = value


def _staged_agent_step(
    agent: Any,
    model: StressModel,
    enabled: set,
    buckets: Dict[str, List[Dict[str, Any]]],
    cfg: Any,
) -> None:
    """Run one day of the enabled blocks for a single agent.

    Mirrors the subevent and daily-consolidation structure of
    ``Person.step()`` but executes only the blocks in ``enabled``.
    """
    state = agent._build_agent_state()
    neighbor_affects = get_neighbor_affects(agent, model)

    if "stress_perception" in enabled or "interaction" in enabled:
        n_subevents = sample_poisson(lam=cfg.get("agent", "subevents_per_day"), rng=agent._rng, min_value=1)
        actions = [agent._rng.choice(["interact", "stress"]) for _ in range(n_subevents)]
        agent._rng.shuffle(actions)

        for action in actions:
            state["support_boost"] = state.get("support_boost", 0.0) * 0.9

            if action == "stress" and "stress_perception" in enabled:
                perception_config = {
                    "omega_c": cfg.get("appraisal", "omega_c"),
                    "omega_o": cfg.get("appraisal", "omega_o"),
                    "bias": cfg.get("appraisal", "bias"),
                    "gamma": cfg.get("appraisal", "gamma"),
                    "delta": 0.2,
                    "base_threshold": cfg.get("threshold", "base_threshold"),
                    "challenge_scale": cfg.get("threshold", "challenge_scale"),
                    "hindrance_scale": cfg.get("threshold", "hindrance_scale"),
                }
                pre = dict(state)
                result = run_stress_perception(state, perception_config, agent._rng)
                state = agent._apply_delta(state, result["state_delta"])
                metric = extract_perception_metrics(result, pre)
                if metric is not None:
                    buckets["perception"].append(metric)

                coped_successfully = True
                if state.get("is_stressed", False) and "resilience_activation" in enabled:
                    activation_config = {
                        "neighbor_affects": neighbor_affects,
                        "base_resource_cost": cfg.get("agent", "resource_cost"),
                    }
                    pre = dict(state)
                    result = run_resilience_activation(state, activation_config, agent._rng)
                    state = agent._apply_delta(state, result["state_delta"])
                    metric = extract_activation_metrics(result, pre)
                    if metric is not None:
                        buckets["activation"].append(metric)
                    coped_successfully = result["observation"].get("coped_successfully", False)

                events = list(state.get("daily_stress_events", []))
                events.append(
                    {
                        "challenge": result["state_delta"].get("challenge", 0.0),
                        "hindrance": result["state_delta"].get("hindrance", 0.0),
                        "is_stressed": state.get("is_stressed", False),
                        "stress_level": state.get("current_stress", 0.0),
                        "coped_successfully": coped_successfully,
                    }
                )
                state["daily_stress_events"] = events

            elif action == "interact" and "interaction" in enabled:
                neighbors = []
                if agent.pos is not None:
                    try:
                        neighbors = list(model.grid.get_neighbors(agent.pos, include_center=False))
                    except Exception:
                        neighbors = []
                if not neighbors:
                    continue
                partner = agent._rng.choice(neighbors)
                partner_state = partner._build_agent_state()
                interaction_config = {
                    "influence_rate": cfg.get("interaction", "influence_rate"),
                    "resilience_influence": cfg.get("interaction", "resilience_influence"),
                }
                pre = dict(state)
                self_output, partner_output = phase_process_interaction(
                    state, partner_state, interaction_config, agent._rng
                )
                metric = extract_interaction_metrics(self_output, pre)
                if metric is not None:
                    buckets["interaction"].append(metric)

                _apply_interaction_delta(state, self_output["state_delta"])
                _apply_interaction_delta(partner_state, partner_output["state_delta"])
                partner._write_back_state(partner_state)

                state["daily_interactions"] = state.get("daily_interactions", 0) + 1
                if self_output["observation"].get("support_occurred", False):
                    state["daily_support_exchanges"] = state.get("daily_support_exchanges", 0) + 1
                    state["support_boost"] = min(1.0, state.get("support_boost", 0.0) + 0.10)

    if "resource_allocation" in enabled:
        resource_config = {
            "base_regeneration": cfg.get("resource", "base_regeneration"),
            "preservable_allocation_fraction": cfg.get("assumptions", "preservable_allocation_fraction"),
            "softmax_temperature": cfg.get("utility", "softmax_temperature"),
            "protective_improvement_rate": cfg.get("resource", "protective_improvement_rate"),
        }
        pre = dict(state)
        result = run_resource_allocation(state, resource_config, agent._rng)
        state = agent._apply_delta(state, result["state_delta"])
        metric = extract_allocation_metrics(result, pre)
        if metric is not None:
            buckets["allocation"].append(metric)

    if "stress_buffering" in enabled:
        pre = dict(state)
        result = run_stress_buffering(state, {}, agent._rng)
        state = agent._apply_delta(state, result["state_delta"])
        metric = extract_buffering_metrics(result, pre)
        if metric is not None:
            buckets["buffering"].append(metric)

    agent._write_back_state(state)


def run_stage(stage_id: int, agents: int, days: int, seed: int) -> StageRun:
    """Run one staged stage (1-6) over a fresh StressModel with the same seed.

    Args:
        stage_id: Stage number in 1..6.
        agents: Population size.
        days: Simulation length in days.
        seed: RNG seed (shared across stages for comparability).

    Returns:
        StageRun with metric buckets, daily summaries, and the final model.

    Raises:
        ValueError: If stage_id is not in 1..6.
    """
    stage = next((s for s in STAGES if s["id"] == stage_id), None)
    if stage is None or stage_id == 7:
        raise ValueError(f"run_stage supports stages 1-6, got {stage_id}")

    cfg = get_config()
    model = StressModel(N=agents, max_days=days, seed=seed)
    buckets = _empty_metric_buckets()

    for agent in model.agents:
        state = agent._build_agent_state()
        buckets["initialization"].append(extract_initialization_metrics(state))

    enabled = set(stage["blocks_active"])
    daily: List[Dict[str, float]] = []
    if stage_id == 1:
        daily.append(_population_summary(model, 0))
    else:
        for day in range(days):
            agents_list = list(model.agents)
            model.random.shuffle(agents_list)
            for agent in agents_list:
                _staged_agent_step(agent, model, enabled, buckets, cfg)
            daily.append(_population_summary(model, day))

    return StageRun(metrics=buckets, daily_summaries=daily, model=model)


def compute_stage_row(stage: Dict[str, Any], agents: int, days: int, run: StageRun) -> Dict[str, Any]:
    """Build one row of cumulative_stages_metrics.csv for a staged stage.

    Args:
        stage: Stage registry entry (stages 1-6).
        agents: Population size.
        days: Simulation length in days.
        run: StageRun from run_stage.

    Returns:
        Row dict with state snapshot plus per-block aggregate metrics.
    """
    row: Dict[str, Any] = {
        "stage": stage["id"],
        "blocks_active": ",".join(stage["blocks_active"]),
        **{col: float("nan") for col in METRIC_COLUMNS},
    }
    agents_list = list(run.model.agents)
    row["avg_pss10"] = float(np.mean([a.pss10 for a in agents_list]))
    row["avg_stress"] = float(np.mean([a.current_stress for a in agents_list]))
    row["avg_affect"] = float(np.mean([a.affect for a in agents_list]))
    row["avg_resilience"] = float(np.mean([a.resilience for a in agents_list]))
    row["avg_resources"] = float(np.mean([a.resources for a in agents_list]))
    row["stress_prevalence"] = float(np.mean([1.0 if a.stressed else 0.0 for a in agents_list]))

    if run.metrics["perception"]:
        agg = aggregate_perception(run.metrics["perception"])
        row["avg_challenge"] = agg.get("challenge_mean", float("nan"))
        row["avg_hindrance"] = agg.get("hindrance_mean", float("nan"))
    row["stress_event_rate"] = len(run.metrics["perception"]) / (days * agents)

    if run.metrics["activation"]:
        agg = aggregate_activation(run.metrics["activation"])
        row["coping_success_rate"] = agg.get("coping_success_rate", float("nan"))

    if run.metrics["interaction"]:
        agg = aggregate_interaction(run.metrics["interaction"])
        row["interactions_per_agent_day"] = len(run.metrics["interaction"]) / (days * agents)
        row["support_exchange_rate"] = agg.get("support_exchange_rate", float("nan"))

    if run.metrics["allocation"]:
        agg = aggregate_allocation(run.metrics["allocation"])
        row["avg_regeneration"] = agg.get("regeneration_mean", float("nan"))
        row["avg_net_resource_change"] = agg.get("net_resource_change_mean", float("nan"))

    if run.metrics["buffering"]:
        agg = aggregate_buffering(run.metrics["buffering"])
        row["avg_buffering_strength"] = agg.get("buffering_strength_mean", float("nan"))
        row["avg_stress_depletion"] = agg.get("stress_depletion_mean", float("nan"))

    return row


def _find_project_root() -> Path:
    """Locate the project root by walking up to simulate.py."""
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "simulate.py").exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError("Could not locate project root (simulate.py missing)")


def run_stage7(
    output_dir: Path,
    agents: int,
    days: int,
    seed: int,
    prefix: str = "stage7",
) -> Tuple[Path, Path]:
    """Run the full model via simulate.py, exporting to output_dir.

    simulate.py writes ``{output_dir}/{prefix}_model.csv`` and
    ``{output_dir}/{prefix}_agent.csv``.

    Args:
        output_dir: Directory for simulate.py outputs.
        agents: Population size.
        days: Simulation length in days.
        seed: RNG seed (matches stages 1-6).
        prefix: Output file prefix for simulate.py.

    Returns:
        Tuple of (model CSV path, agent CSV path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "simulate.py",
        "--days",
        str(days),
        "--seed",
        str(seed),
        "--agents",
        str(agents),
        "--prefix",
        prefix,
        "--output-data",
        str(output_dir),
    ]
    subprocess.run(cmd, cwd=_find_project_root(), check=True)
    return output_dir / f"{prefix}_model.csv", output_dir / f"{prefix}_agent.csv"


def compute_stage7_row(model_csv: Path, agents: int, days: int, stage: Dict[str, Any]) -> Dict[str, Any]:
    """Build the stage 7 row from simulate.py's model CSV.

    State columns come from the final day; phase columns are run means.
    Allocation and buffering metrics are read from the model-level CSV
    (exported via MODEL_REPORTERS).

    Args:
        model_csv: Path to simulate.py model-level CSV.
        agents: Population size.
        days: Simulation length in days (unused, kept for signature symmetry).
        stage: Stage registry entry (stage 7).

    Returns:
        Row dict compatible with compute_stage_row.
    """
    model_df = pd.read_csv(model_csv)
    row: Dict[str, Any] = {
        "stage": stage["id"],
        "blocks_active": ",".join(stage["blocks_active"]),
        **{col: float("nan") for col in METRIC_COLUMNS},
    }
    last = model_df.iloc[-1]
    for col in ("avg_pss10", "avg_stress", "avg_affect", "avg_resilience", "avg_resources", "stress_prevalence"):
        if col in model_df.columns:
            row[col] = float(last[col])
    for col in ("avg_challenge", "avg_hindrance", "coping_success_rate"):
        if col in model_df.columns:
            row[col] = float(model_df[col].mean())
    if "stress_events" in model_df.columns:
        row["stress_event_rate"] = float(model_df["stress_events"].mean()) / agents
    if "social_interactions" in model_df.columns:
        row["interactions_per_agent_day"] = float(model_df["social_interactions"].mean()) / agents
    if "daily_social_support_rate" in model_df.columns:
        row["support_exchange_rate"] = float(model_df["daily_social_support_rate"].mean())
    for col in ("avg_regeneration", "avg_buffering_strength"):
        if col in model_df.columns:
            row[col] = float(model_df[col].mean())
    return row


def _summarize(series: pd.Series) -> Dict[str, float]:
    """Return mean, sample sd, CV (%), and min/max for a numeric series.

    Args:
        series: Numeric pandas Series.

    Returns:
        Dict with keys mean, sd, cv, min, max. CV is NaN when mean is zero.
    """
    mean = float(series.mean())
    sd = float(series.std(ddof=1))
    cv = float(sd / abs(mean) * 100.0) if mean != 0 else float("nan")
    return {"mean": mean, "sd": sd, "cv": cv, "min": float(series.min()), "max": float(series.max())}


def _stats_rows(level: str, df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
    """Build tidy stats rows for the columns present in df.

    Args:
        level: Stats level label (population, agent_step, final_day).
        df: Dataframe to summarize.
        columns: Candidate metric columns; missing ones are skipped.

    Returns:
        List of row dicts with level, metric, mean, sd, cv, min, max.
    """
    rows = []
    for col in columns:
        if col in df.columns:
            rows.append({"level": level, "metric": col, **_summarize(df[col])})
    return rows


def _render_population_figure(df: pd.DataFrame, figures_dir: Path, filename: str) -> Optional[str]:
    """Render a population-state PDF when all required columns are present.

    Args:
        df: Agent-slice dataframe (current_stress renamed to stress).
        figures_dir: Directory for the PDF.
        filename: Output filename.

    Returns:
        Path to the generated PDF, or None if required columns are missing.
    """
    required = ["resilience", "affect", "stress", "pss10"]
    if not set(required).issubset(df.columns):
        return None
    return create_visualization_report(df, str(figures_dir), filename)


def export_results_assets(
    model_csv: Union[str, Path],
    agent_csv: Union[str, Path],
    output_dir: Union[str, Path] = "data/output",
    figures_dir: Union[str, Path] = "docs/figures",
    initial_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Export article-ready figures and summary stats from stage-7 exports.

    Renders the initial-population, final-population, and time-series figures
    and writes stage7_results_stats.csv (tidy long format with levels
    population, agent_step, final_day, and correlation rows). The initial
    figure and the initial pss10-stress correlation use the pre-step baseline
    (simulate.py's {prefix}_initial.csv) when available, falling back to the
    Step==1 slice of the agent CSV.

    Args:
        model_csv: Path to the stage-7 model-level CSV (simulate.py export).
        agent_csv: Path to the stage-7 agent-level CSV (simulate.py export).
        output_dir: Directory for the stats CSV (created if missing).
        figures_dir: Directory for rendered PDF figures (created if missing).
        initial_csv: Optional path to the pre-step initial agent CSV.

    Returns:
        Dict mapping asset names to output paths (empty string when a figure
        could not be rendered because its required columns were absent).
    """
    model_df = pd.read_csv(model_csv)
    agent_df = pd.read_csv(agent_csv)
    out = Path(output_dir)
    figs = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    step_col = "Step"
    first_step = int(agent_df[step_col].min())
    last_step = int(agent_df[step_col].max())
    rename_map = {"current_stress": "stress"}
    if initial_csv is not None and Path(initial_csv).exists():
        initial_df = pd.read_csv(initial_csv)
    else:
        initial_df = agent_df[agent_df[step_col] == first_step]
    initial_fig = _render_population_figure(
        initial_df.rename(columns=rename_map),
        figs,
        FIGURE_FILENAMES["initial_population"],
    )
    final_fig = _render_population_figure(
        agent_df[agent_df[step_col] == last_step].rename(columns=rename_map),
        figs,
        FIGURE_FILENAMES["final_population"],
    )
    ts_required = ["avg_pss10", "avg_stress", "avg_resilience", "avg_affect"]
    ts_fig = (
        create_time_series_visualization(model_df, str(figs), FIGURE_FILENAMES["time_series"])
        if set(ts_required).issubset(model_df.columns)
        else None
    )

    rows = _stats_rows("population", model_df, MODEL_STAT_COLUMNS)
    rows += _stats_rows("agent_step", agent_df, AGENT_STAT_COLUMNS)
    rows += _stats_rows("final_day", agent_df[agent_df[step_col] == last_step], AGENT_STAT_COLUMNS)
    if {"pss10", "current_stress"}.issubset(agent_df.columns):
        initial_corr_df = initial_df if {"pss10", "current_stress"}.issubset(initial_df.columns) else None
        corr_slices = [
            ("pss10_stress_initial", initial_corr_df),
            ("pss10_stress_final", agent_df[agent_df[step_col] == last_step]),
        ]
        for label, slice_df in corr_slices:
            if slice_df is None or len(slice_df) == 0:
                continue
            r = float(slice_df["pss10"].corr(slice_df["current_stress"]))
            rows.append(
                {
                    "level": "correlation",
                    "metric": label,
                    "mean": r,
                    "sd": float("nan"),
                    "cv": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            )

    stats_csv = out / "stage7_results_stats.csv"
    pd.DataFrame(rows, columns=["level", "metric", "mean", "sd", "cv", "min", "max"]).to_csv(stats_csv, index=False)

    return {
        "initial_population": str(initial_fig) if initial_fig else "",
        "final_population": str(final_fig) if final_fig else "",
        "time_series": str(ts_fig) if ts_fig else "",
        "stats_csv": str(stats_csv),
    }


def run_cumulative_demo(
    agents: int = DEFAULT_AGENTS,
    days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
    output_dir: str = "data/output",
    figures_dir: str = "docs/figures",
) -> Path:
    """Run all 7 stages and export results to output_dir.

    Args:
        agents: Population size.
        days: Simulation length in days.
        seed: RNG seed shared across stages.
        output_dir: Output directory (created if missing).
        figures_dir: Directory for stage-7 figures and stats (created if missing).

    Returns:
        Path of the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    model_csv: Optional[Path] = None
    agent_csv: Optional[Path] = None

    for stage in STAGES:
        if stage["id"] == 7:
            model_csv, agent_csv = run_stage7(out, agents, days, seed)
            rows.append(compute_stage7_row(model_csv, agents, days, stage))
        else:
            run = run_stage(stage["id"], agents, days, seed)
            rows.append(compute_stage_row(stage, agents, days, run))
            pd.DataFrame(run.daily_summaries).to_csv(out / f"cumulative_stage{stage['id']}_summary.csv", index=False)

    pd.DataFrame(rows).to_csv(out / "cumulative_stages_metrics.csv", index=False)
    metadata = {
        "seed": seed,
        "agents": agents,
        "days": days,
        "stages": [{"id": s["id"], "name": s["name"], "blocks_active": s["blocks_active"]} for s in STAGES],
    }
    (out / "cumulative_blocks.json").write_text(json.dumps(metadata, indent=2))
    if model_csv is not None and agent_csv is not None:
        initial_csv = agent_csv.with_name(agent_csv.name.replace("_agent.csv", "_initial.csv"))
        export_results_assets(model_csv, agent_csv, str(out), figures_dir, initial_csv)
    return out


def main(argv: Optional[list] = None) -> int:
    """CLI entry point for the cumulative block demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the 7-stage cumulative block demo.")
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS, help="population size")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="simulation length in days")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed shared across stages")
    parser.add_argument("--output-dir", default="data/output", help="output directory")
    parser.add_argument("--figures-dir", default="docs/figures", help="figure and stats directory")
    args = parser.parse_args(argv)
    out = run_cumulative_demo(args.agents, args.days, args.seed, args.output_dir, args.figures_dir)
    print(f"Cumulative block demo written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
