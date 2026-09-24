"""
Pytest configuration and shared fixtures for ABM simulation tests.
"""

import logging
import os

import numpy as np
import networkx as nx
import pytest

from src.python.assumption_config import reload_assumptions
from src.python.config import get_config, reload_config, config
from src.python.math_utils import create_rng
from src.python.stress_utils import generate_stress_event
from src.python.agent import Person
from src.python.affect_utils import ProtectiveFactors, InteractionConfig, ResourceParams
from src.python.stress_utils import ThresholdParams, AppraisalWeights
from src.python.phases.interfaces import AgentState

# Initialize assumption config cache at import time
reload_assumptions()

# Env vars that can leak between tests (from _apply_overrides, Config() in temp dirs, etc.)
_LEAKING_ENV_KEYS = [
    "ASSUMPTION_RESOURCE_PENALTY",
    "ASSUMPTION_FAILED_COPING_COST_PENALTY",
    "ASSUMPTION_AFFECT_DETERIORATION_SCALE",
    "ASSUMPTION_AFFECT_REGENERATION_MULTIPLIER",
    "ASSUMPTION_RESILIENCE_IMPROVEMENT_SCALE",
    "ASSUMPTION_COPING_SOCIAL_SUPPORT_FACTOR",
    "ASSUMPTION_COPING_SUPPORT_BOOST_FACTOR",
    "ASSUMPTION_RESILIENCE_COPING_FACTOR",
    "ASSUMPTION_CONTROLLABILITY_HOMEOSTASIS_RATE",
    "ASSUMPTION_OVERLOAD_HOMEOSTASIS_RATE",
    "PSS10_RESILIENCE_COUPLING",
    "PSS10_BIAS_SD",
    "PSS10_STRESS_DAMPENING",
    "PSS10_NOISE_SD",
    "PSS10_ITEM_MEAN",
    "PSS10_ITEM_SD",
    "PSS10_SCALE",
    "PSS10_BIFACTOR_COR",
    "PSS10_LOAD_CONTROLLABILITY",
    "PSS10_SKEW_A",
    "PSS10_THRESHOLD",
    "PSS10_CONTROLLABILITY_SD",
    "PSS10_OVERLOAD_SD",
    "PSS10_ESTIMATION_BASE",
    "STRESS_DELTA",
]

## Phase-specific fixtures


@pytest.fixture
def phase_minimal_state():
    """Minimal AgentState for phase function tests."""
    return AgentState(
        resilience=0.5,
        affect=0.0,
        resources=0.6,
        baseline_resilience=0.5,
        baseline_affect=0.0,
        current_stress=0.3,
        protective_factors={
            "social_support": 0.5,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        },
        pss10=15,
        stressed=False,
        daily_interactions=0,
        daily_support_exchanges=0,
        stress_controllability=0.5,
        stress_overload=0.5,
        consecutive_hindrances=0.0,
        volatility=0.3,
        stress_config={},
        interaction_config={},
    )


@pytest.fixture
def phase_config():
    """Generic phase configuration dict."""
    return {
        "threshold": 0.5,
        "influence_rate": 0.1,
        "decay_rate": 0.05,
    }


@pytest.fixture
def config():
    """Provide configuration instance for tests."""
    return get_config()


@pytest.fixture
def sample_rng():
    """Provide a seeded random number generator for reproducible tests."""
    return create_rng(42)


@pytest.fixture
def sample_stress_event(sample_rng):
    """Provide a sample stress event for testing."""
    return generate_stress_event(sample_rng)


@pytest.fixture
def clean_env():
    """Fixture to ensure clean environment variables for testing."""
    # Store original environment
    original_env = dict(os.environ)

    # Clear test-specific variables
    test_vars = [
        "SIMULATION_NUM_AGENTS",
        "AGENT_INITIAL_RESILIENCE",
        "NETWORK_WATTS_K",
        "INTERACTION_INFLUENCE_RATE",
        "PROTECTIVE_SOCIAL_SUPPORT",
        "RESOURCE_BASE_REGENERATION",
        "APPRAISAL_OMEGA_C",
        "THRESHOLD_BASE_THRESHOLD",
        "OUTPUT_SAVE_TIME_SERIES",
        "OUTPUT_SAVE_NETWORK_SNAPSHOTS",
    ]

    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def reload_config_fixture():
    """Fixture to reload configuration after environment changes."""

    def _reload():
        return reload_config()

    return _reload


@pytest.fixture
def sample_agents():
    """Provide sample agent instances for testing."""
    # Create mock model
    model = Mock()
    model.seed = 42
    model.grid = Mock()
    model.grid.get_neighbors.return_value = []
    model.agents = Mock()
    model.register_agent = Mock()
    model.rng = np.random.default_rng(42)

    # Create sample agents with different configurations
    agent1 = Person(model, {"initial_resilience": 0.8, "initial_affect": 0.2})
    agent2 = Person(model, {"initial_resilience": 0.3, "initial_affect": -0.5})
    agent3 = Person(model, {"initial_resilience": 0.6, "initial_affect": 0.0})

    return [agent1, agent2, agent3]


@pytest.fixture
def sample_protective_factors():
    """Provide sample protective factors for testing."""
    return ProtectiveFactors(social_support=0.7, family_support=0.5, formal_intervention=0.3, psychological_capital=0.8)


@pytest.fixture
def sample_interaction_config():
    """Provide sample interaction configuration for testing."""
    return InteractionConfig(influence_rate=0.1, resilience_influence=0.05, max_neighbors=10)


@pytest.fixture
def sample_threshold_params():
    """Provide sample threshold parameters for testing."""
    return ThresholdParams(base_threshold=0.5, challenge_scale=0.15, hindrance_scale=0.25)


@pytest.fixture
def sample_appraisal_weights():
    """Provide sample appraisal weights for testing."""
    return AppraisalWeights(omega_c=1.0, omega_p=1.0, omega_o=1.0, bias=0.0, gamma=6.0)


@pytest.fixture
def sample_resource_params():
    """Provide sample resource parameters for testing."""
    return ResourceParams(base_regeneration=0.1, allocation_cost=0.15, cost_exponent=1.5)


@pytest.fixture
def deterministic_rng():
    """Provide a deterministic RNG for reproducible tests."""
    return np.random.default_rng(12345)


@pytest.fixture
def test_data_directory(tmp_path):
    """Provide a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_time_series_data():
    """Provide sample time series data for testing."""
    # Generate sample time series data for testing
    time_steps = np.arange(0, 100, 1)
    resilience_data = 0.5 + 0.3 * np.sin(0.1 * time_steps) + 0.1 * np.random.randn(100)
    affect_data = 0.2 * np.cos(0.15 * time_steps) + 0.05 * np.random.randn(100)

    # Clamp to valid ranges
    resilience_data = np.clip(resilience_data, 0, 1)
    affect_data = np.clip(affect_data, -1, 1)

    return {"time_steps": time_steps, "resilience": resilience_data, "affect": affect_data}


@pytest.fixture
def sample_network_data():
    """Provide sample network data for testing."""
    # Create a small test network
    G = nx.watts_strogatz_graph(20, k=4, p=0.1, seed=42)

    # Add some node attributes
    for node in G.nodes():
        G.nodes[node]["resilience"] = np.random.beta(2, 2)
        G.nodes[node]["affect"] = np.random.uniform(-1, 1)

    return G


@pytest.fixture
def benchmark_config():
    """Provide configuration for benchmark testing."""
    return {
        "num_runs": 10,
        "num_agents": 100,
        "time_steps": 365,
        "stress_probability": 0.1,
        "network_type": "watts_strogatz",
        "network_params": {"k": 6, "p": 0.1},
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Automatically set up test logging for all tests."""
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors during tests
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    yield

    # Clean up logging after test
    logging.getLogger().handlers.clear()


# Start with a clean env vars for each run
@pytest.fixture(autouse=True)
def complete_env_isolation():
    """Complete environment isolation for each test.

    Clears env at setup so each test starts with config defaults.
    Saves env before clearing and restores after.
    """

    # SETUP: Clear environment for proper isolation
    current_env = dict(os.environ)
    os.environ.clear()

    # Pin DOTENV_FILE to the repo's empty env file so reload_config() never
    # falls back to a developer's local .env (which may carry stale values
    # that diverge from code defaults and leak into get_assumptions()).
    os.environ["DOTENV_FILE"] = ".env.empty"

    # Set explicit defaults for Fix 3/4 to prevent env-var pollution
    os.environ["APPRAISAL_GAMMA"] = "3.0"
    os.environ["ASSUMPTION_STRESS_DECAY_RATE"] = "0.08"
    os.environ["ASSUMPTION_PF_ALLOCATION_FRACTION"] = "0.05"

    from src.python.config import reload_config as _reload_config

    _reload_config()
    reload_assumptions()

    yield

    # TEARDOWN: Restore environment to pre-test state
    os.environ.clear()
    os.environ.update(current_env)
    _reload_config()
    reload_assumptions()
    # Reload phase modules that cache assumption constants at import time
    import importlib
    import src.python.phases.resilience_activation as _ra

    importlib.reload(_ra)


def run_stress_cycle(agent):
    """Run one stress perception + (if stressed) resilience activation cycle.

    Replaces the deleted Person.stressful_event() for test use.
    Generates an event, appraises it, and handles both stressed/non-stressed paths.

    Returns (challenge, hindrance) tuple.
    """
    from src.python.config import get_config
    from src.python.stress_utils import (
        generate_stress_event,
        process_stress_event,
        AppraisalWeights,
        ThresholdParams,
        update_stress_dimensions_from_event,
    )
    from src.python.affect_utils import get_neighbor_affects
    from src.python.phases.resilience_activation import run_phase as run_resilience_activation

    cfg = get_config()
    event = generate_stress_event(rng=agent._rng)
    weights = AppraisalWeights(
        omega_c=cfg.get("appraisal", "omega_c"),
        omega_o=cfg.get("appraisal", "omega_o"),
        bias=cfg.get("appraisal", "bias"),
        gamma=cfg.get("appraisal", "gamma"),
    )
    threshold_params = ThresholdParams(
        base_threshold=cfg.get("threshold", "base_threshold"),
        challenge_scale=cfg.get("threshold", "challenge_scale"),
        hindrance_scale=cfg.get("threshold", "hindrance_scale"),
    )
    is_stressed, challenge, hindrance = process_stress_event(event, threshold_params, weights, rng=agent._rng)
    state = agent._build_agent_state()
    state.update(
        {
            "challenge": challenge,
            "hindrance": hindrance,
            "is_stressed": is_stressed,
            "event_controllability": event.controllability,
            "event_overload": event.overload,
        }
    )
    if not is_stressed:
        (
            state["stress_controllability"],
            state["stress_overload"],
            state["recent_stress_intensity"],
            state["stress_momentum"],
        ) = update_stress_dimensions_from_event(
            current_controllability=state["stress_controllability"],
            current_overload=state["stress_overload"],
            challenge=challenge,
            hindrance=hindrance,
            coped_successfully=True,
            is_stressful=False,
            volatility=state["volatility"],
            recent_stress_intensity=state["recent_stress_intensity"],
            stress_momentum=state["stress_momentum"],
        )
        agent._write_back_state(state)
        return challenge, hindrance
    neighbor_affects = get_neighbor_affects(agent, agent.model)
    activation_result = run_resilience_activation(
        state,
        {
            "neighbor_affects": neighbor_affects,
            "base_resource_cost": cfg.get("agent", "resource_cost"),
        },
        agent._rng,
    )
    state = agent._apply_delta(state, activation_result["state_delta"])
    pss10 = state.get("pss10", 0)
    if pss10 > 0:
        scores = list(state.get("daily_pss10_scores", []))
        scores.append(pss10)
        state["daily_pss10_scores"] = scores
    agent._write_back_state(state)
    agent.daily_stress_events.append(
        {
            "challenge": challenge,
            "hindrance": hindrance,
            "is_stressed": is_stressed,
            "stress_level": state.get("current_stress", 0.0),
            "coped_successfully": activation_result["observation"].get("coped_successfully", False),
            "event_controllability": event.controllability,
            "event_overload": event.overload,
        }
    )
    return challenge, hindrance
