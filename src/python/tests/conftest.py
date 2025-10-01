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