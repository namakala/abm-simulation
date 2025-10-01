"""
Pytest configuration and shared fixtures for ABM simulation tests.
"""

import os
import pytest
from src.python.config import get_config, reload_config


@pytest.fixture
def config():
    """Provide configuration instance for tests."""
    return get_config()


@pytest.fixture
def sample_rng():
    """Provide a seeded random number generator for reproducible tests."""
    from src.python.math_utils import create_rng
    return create_rng(42)


@pytest.fixture
def sample_stress_event(sample_rng):
    """Provide a sample stress event for testing."""
    from src.python.stress_utils import generate_stress_event
    return generate_stress_event(sample_rng)


@pytest.fixture
def clean_env():
    """Fixture to ensure clean environment variables for testing."""
    # Store original environment
    original_env = dict(os.environ)

    # Clear test-specific variables
    test_vars = [
        'SIMULATION_NUM_AGENTS', 'AGENT_INITIAL_RESILIENCE',
        'NETWORK_WATTS_K', 'INTERACTION_INFLUENCE_RATE',
        'PROTECTIVE_SOCIAL_SUPPORT', 'RESOURCE_BASE_REGENERATION',
        'APPRAISAL_OMEGA_C', 'THRESHOLD_BASE_THRESHOLD',
        'OUTPUT_SAVE_TIME_SERIES', 'OUTPUT_SAVE_NETWORK_SNAPSHOTS'
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
    from src.python.agent import Person
    from unittest.mock import Mock

    # Create mock model
    model = Mock()
    model.seed = 42
    model.grid = Mock()
    model.grid.get_neighbors.return_value = []
    model.agents = Mock()
    model.register_agent = Mock()
    model.rng = np.random.default_rng(42)

    # Create sample agents with different configurations
    agent1 = Person(model, {'initial_resilience': 0.8, 'initial_affect': 0.2})
    agent2 = Person(model, {'initial_resilience': 0.3, 'initial_affect': -0.5})
    agent3 = Person(model, {'initial_resilience': 0.6, 'initial_affect': 0.0})

    return [agent1, agent2, agent3]


@pytest.fixture
def sample_protective_factors():
    """Provide sample protective factors for testing."""
    from src.python.affect_utils import ProtectiveFactors

    return ProtectiveFactors(
        social_support=0.7,
        family_support=0.5,
        formal_intervention=0.3,
        psychological_capital=0.8
    )


@pytest.fixture
def sample_interaction_config():
    """Provide sample interaction configuration for testing."""
    from src.python.affect_utils import InteractionConfig

    return InteractionConfig(
        influence_rate=0.1,
        resilience_influence=0.05,
        max_neighbors=10
    )


@pytest.fixture
def sample_threshold_params():
    """Provide sample threshold parameters for testing."""
    from src.python.stress_utils import ThresholdParams

    return ThresholdParams(
        base_threshold=0.5,
        challenge_scale=0.15,
        hindrance_scale=0.25
    )


@pytest.fixture
def sample_appraisal_weights():
    """Provide sample appraisal weights for testing."""
    from src.python.stress_utils import AppraisalWeights

    return AppraisalWeights(
        omega_c=1.0,
        omega_p=1.0,
        omega_o=1.0,
        bias=0.0,
        gamma=6.0
    )


@pytest.fixture
def sample_resource_params():
    """Provide sample resource parameters for testing."""
    from src.python.affect_utils import ResourceParams

    return ResourceParams(
        base_regeneration=0.1,
        allocation_cost=0.15,
        cost_exponent=1.5
    )


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

    return {
        'time_steps': time_steps,
        'resilience': resilience_data,
        'affect': affect_data
    }


@pytest.fixture
def sample_network_data():
    """Provide sample network data for testing."""
    import networkx as nx

    # Create a small test network
    G = nx.watts_strogatz_graph(20, k=4, p=0.1, seed=42)

    # Add some node attributes
    for node in G.nodes():
        G.nodes[node]['resilience'] = np.random.beta(2, 2)
        G.nodes[node]['affect'] = np.random.uniform(-1, 1)

    return G


@pytest.fixture
def benchmark_config():
    """Provide configuration for benchmark testing."""
    return {
        'num_runs': 10,
        'num_agents': 100,
        'time_steps': 365,
        'stress_probability': 0.1,
        'network_type': 'watts_strogatz',
        'network_params': {'k': 6, 'p': 0.1}
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Automatically set up test logging for all tests."""
    import logging

    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    yield

    # Clean up logging after test
    logging.getLogger().handlers.clear()