"""
Comprehensive test suite for visualization_utils.py module.

This module tests the create_visualization_report function with various scenarios,
including edge cases, matplotlib availability, and integration with simulation data.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

# Import the function and constants
from src.python.visualization_utils import create_visualization_report, HAS_MATPLOTLIB, HAS_SCIPY


class MockAgent:
    """Mock agent for testing visualization functions."""

    def __init__(self, resilience=0.5, affect=0.0, resources=0.5, pss10=20):
        self.resilience = resilience
        self.affect = affect
        self.resources = resources
        self.pss10 = pss10


class TestVisualizationUtils:
    """Test class for visualization utilities."""

    @pytest.fixture
    def temp_dir(self):
        """Provide temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_agents(self):
        """Provide sample agents for testing."""
        return [
            MockAgent(resilience=0.6, affect=0.1, resources=0.7, pss10=15),
            MockAgent(resilience=0.4, affect=-0.2, resources=0.5, pss10=25),
            MockAgent(resilience=0.8, affect=0.3, resources=0.9, pss10=10),
            MockAgent(resilience=0.2, affect=-0.1, resources=0.3, pss10=30),
            MockAgent(resilience=0.5, affect=0.0, resources=0.6, pss10=20)
        ]

    def test_create_visualization_with_matplotlib(self, sample_agents, temp_dir):
        """Test visualization generation when matplotlib is available."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        viz_path = create_visualization_report(sample_agents, temp_dir, "test_plot.png")

        # Verify file was created
        assert Path(viz_path).exists()
        assert viz_path.endswith("test_plot.png")

        # Verify it's in the correct directory
        assert temp_dir in viz_path

        # Verify file is a valid image (basic check)
        assert Path(viz_path).stat().st_size > 0

    def test_create_visualization_without_matplotlib(self, sample_agents, temp_dir):
        """Test visualization when matplotlib is not available."""
        # Mock matplotlib as unavailable
        with patch('src.python.visualization_utils.HAS_MATPLOTLIB', False):
            viz_path = create_visualization_report(sample_agents, temp_dir, "test_plot.png")

            # Should return placeholder file path
            assert viz_path.endswith("visualization_not_available.txt")
            assert Path(viz_path).exists()

            # Verify placeholder content
            with open(viz_path, 'r') as f:
                content = f.read()
                assert "matplotlib" in content.lower() or "not available" in content.lower()

    def test_empty_agents_list(self, temp_dir):
        """Test visualization with empty agents list."""
        empty_agents = []

        viz_path = create_visualization_report(empty_agents, temp_dir, "empty_plot.png")

        if HAS_MATPLOTLIB:
            # Should still generate a plot (empty histograms)
            assert Path(viz_path).exists()
        else:
            # Should return placeholder
            assert viz_path.endswith("visualization_not_available.txt")

    def test_custom_filename(self, sample_agents, temp_dir):
        """Test visualization with custom filename."""
        custom_filename = "custom_analysis.png"

        viz_path = create_visualization_report(sample_agents, temp_dir, custom_filename)

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()
            assert custom_filename in viz_path

    def test_default_filename(self, sample_agents, temp_dir):
        """Test visualization with default filename."""
        viz_path = create_visualization_report(sample_agents, temp_dir)

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()
            # Should contain population size in filename
            assert str(len(sample_agents)) in Path(viz_path).name

    def test_output_directory_creation(self, sample_agents):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as base_dir:
            nested_dir = Path(base_dir) / "nested" / "deep" / "path"
            viz_path = create_visualization_report(sample_agents, str(nested_dir), "test.png")

            if HAS_MATPLOTLIB:
                assert Path(viz_path).exists()
                assert nested_dir.exists()

    def test_agents_with_extreme_values(self, temp_dir):
        """Test visualization with agents having extreme attribute values."""
        extreme_agents = [
            MockAgent(resilience=0.0, affect=-1.0, resources=0.0, pss10=0),
            MockAgent(resilience=1.0, affect=1.0, resources=1.0, pss10=40),
            MockAgent(resilience=0.5, affect=0.0, resources=0.5, pss10=20)
        ]

        viz_path = create_visualization_report(extreme_agents, temp_dir, "extreme_plot.png")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_agents_with_identical_values(self, temp_dir):
        """Test visualization with agents having identical attribute values."""
        identical_agents = [
            MockAgent(resilience=0.5, affect=0.0, resources=0.5, pss10=20),
            MockAgent(resilience=0.5, affect=0.0, resources=0.5, pss10=20),
            MockAgent(resilience=0.5, affect=0.0, resources=0.5, pss10=20)
        ]

        viz_path = create_visualization_report(identical_agents, temp_dir, "identical_plot.png")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_large_number_of_agents(self, temp_dir):
        """Test visualization with a large number of agents."""
        large_agents = [MockAgent() for _ in range(1000)]

        viz_path = create_visualization_report(large_agents, temp_dir, "large_plot.png")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_scipy_availability_for_qq_plot(self, sample_agents, temp_dir):
        """Test Q-Q plot generation based on scipy availability."""
        viz_path = create_visualization_report(sample_agents, temp_dir, "qq_test.png")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()
            # If scipy is available, Q-Q plot should be generated
            # If not, it should still work but without Q-Q plot

    def test_visualization_content_validation(self, sample_agents, temp_dir):
        """Test that visualization contains expected elements."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        viz_path = create_visualization_report(sample_agents, temp_dir, "content_test.png")

        # Basic validation that file is not empty
        assert Path(viz_path).stat().st_size > 1000  # Reasonable size for a plot

    def test_error_handling_invalid_agents(self, temp_dir):
        """Test error handling with invalid agent data."""
        # Agents with invalid attributes
        invalid_agents = [
            MockAgent(resilience="invalid", affect=0.0, resources=0.5, pss10=20),
            MockAgent(resilience=0.5, affect=None, resources=0.5, pss10=20)
        ]

        # Should handle gracefully without crashing
        viz_path = create_visualization_report(invalid_agents, temp_dir, "invalid_test.png")

        if HAS_MATPLOTLIB:
            # Should still generate something or handle error
            assert isinstance(viz_path, str)

    def test_integration_with_simulation_data(self, temp_dir):
        """Test integration with actual simulation data structure."""
        # Mock agents that mimic real Person agents
        class MockPersonAgent(MockAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Add additional attributes that real agents might have
                self.unique_id = np.random.randint(1000)
                self.pos = (np.random.random(), np.random.random())

        sim_agents = [MockPersonAgent() for _ in range(50)]

        viz_path = create_visualization_report(sim_agents, temp_dir, "simulation_integration.png")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_reproducibility_of_visualization(self, sample_agents, temp_dir):
        """Test that visualizations are reproducible with same inputs."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        # Generate two visualizations with same inputs
        viz1 = create_visualization_report(sample_agents, temp_dir, "repro1.png")
        viz2 = create_visualization_report(sample_agents, temp_dir, "repro2.png")

        # Files should both exist and have similar sizes (approximately)
        assert Path(viz1).exists()
        assert Path(viz2).exists()

        size1 = Path(viz1).stat().st_size
        size2 = Path(viz2).stat().st_size

        # Allow some variation but should be reasonably close
        assert abs(size1 - size2) / max(size1, size2) < 0.1  # Within 10%

    def test_visualization_with_zero_variance(self, temp_dir):
        """Test visualization when all agents have identical values."""
        zero_var_agents = [
            MockAgent(resilience=0.5, affect=0.0, resources=0.5, pss10=20)
        ] * 10

        viz_path = create_visualization_report(zero_var_agents, temp_dir, "zero_var.png")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_visualization_with_boundary_values(self, temp_dir):
        """Test visualization with agents at boundary values."""
        boundary_agents = [
            MockAgent(resilience=0.0, affect=-1.0, resources=0.0, pss10=0),
            MockAgent(resilience=1.0, affect=1.0, resources=1.0, pss10=40)
        ]

        viz_path = create_visualization_report(boundary_agents, temp_dir, "boundary.png")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()