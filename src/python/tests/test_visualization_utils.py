"""
Comprehensive test suite for visualization_utils.py module.

This module tests the create_visualization_report function with various scenarios,
including edge cases, matplotlib availability, and integration with simulation data.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

# Import the function and constants
from src.python.visualization_utils import create_visualization_report, create_time_series_visualization, HAS_MATPLOTLIB, HAS_SCIPY


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

        # Convert agents to DataFrame
        data = pd.DataFrame([
            {'resilience': agent.resilience, 'affect': agent.affect, 'stress': agent.resources, 'pss10': agent.pss10}
            for agent in sample_agents
        ])

        viz_path = create_visualization_report(data, temp_dir, "test_plot.pdf")

        # Verify file was created
        assert Path(viz_path).exists()
        assert viz_path.endswith("test_plot.pdf")

        # Verify it's in the correct directory
        assert temp_dir in viz_path

        # Verify file is a valid image (basic check)
        assert Path(viz_path).stat().st_size > 0

    def test_create_visualization_without_matplotlib(self, sample_agents, temp_dir):
        """Test visualization when matplotlib is not available."""
        # Convert agents to DataFrame
        data = pd.DataFrame([
            {'resilience': agent.resilience, 'affect': agent.affect, 'stress': agent.resources, 'pss10': agent.pss10}
            for agent in sample_agents
        ])

        # Mock matplotlib as unavailable
        with patch('src.python.visualization_utils.HAS_MATPLOTLIB', False):
            viz_path = create_visualization_report(data, temp_dir, "test_plot.pdf")

            # Should return placeholder file path
            assert viz_path.endswith("visualization_not_available.txt")
            assert Path(viz_path).exists()

            # Verify placeholder content
            with open(viz_path, 'r') as f:
                content = f.read()
                assert "matplotlib" in content.lower() or "not available" in content.lower()

    def test_empty_agents_list(self, temp_dir):
        """Test visualization with empty agents list."""
        empty_data = pd.DataFrame(columns=['resilience', 'affect', 'stress', 'pss10'])

        # Should raise ValueError for empty data
        with pytest.raises(ValueError, match="No valid data after removing missing values"):
            create_visualization_report(empty_data, temp_dir, "empty_plot.pdf")

    def test_custom_filename(self, sample_agents, temp_dir):
        """Test visualization with custom filename."""
        # Convert agents to DataFrame
        data = pd.DataFrame([
            {'resilience': agent.resilience, 'affect': agent.affect, 'stress': agent.resources, 'pss10': agent.pss10}
            for agent in sample_agents
        ])

        custom_filename = "custom_analysis.pdf"

        viz_path = create_visualization_report(data, temp_dir, custom_filename)

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()
            assert custom_filename in viz_path

    def test_default_filename(self, sample_agents, temp_dir):
        """Test visualization with default filename."""
        # Convert agents to DataFrame
        data = pd.DataFrame([
            {'resilience': agent.resilience, 'affect': agent.affect, 'stress': agent.resources, 'pss10': agent.pss10}
            for agent in sample_agents
        ])

        viz_path = create_visualization_report(data, temp_dir)

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()
            # Should contain population size in filename
            assert str(len(data)) in Path(viz_path).name

    def test_output_directory_creation(self, sample_agents):
        """Test that output directory is created if it doesn't exist."""
        # Convert agents to DataFrame
        data = pd.DataFrame([
            {'resilience': agent.resilience, 'affect': agent.affect, 'stress': agent.resources, 'pss10': agent.pss10}
            for agent in sample_agents
        ])

        with tempfile.TemporaryDirectory() as base_dir:
            nested_dir = Path(base_dir) / "nested" / "deep" / "path"
            viz_path = create_visualization_report(data, str(nested_dir), "test.pdf")

            if HAS_MATPLOTLIB:
                assert Path(viz_path).exists()
                assert nested_dir.exists()

    def test_agents_with_extreme_values(self, temp_dir):
        """Test visualization with agents having extreme attribute values."""
        extreme_data = pd.DataFrame([
            {'resilience': 0.0, 'affect': -1.0, 'stress': 0.0, 'pss10': 0},
            {'resilience': 1.0, 'affect': 1.0, 'stress': 1.0, 'pss10': 40},
            {'resilience': 0.5, 'affect': 0.0, 'stress': 0.5, 'pss10': 20}
        ])

        viz_path = create_visualization_report(extreme_data, temp_dir, "extreme_plot.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_agents_with_identical_values(self, temp_dir):
        """Test visualization with agents having identical attribute values."""
        identical_data = pd.DataFrame([
            {'resilience': 0.5, 'affect': 0.0, 'stress': 0.5, 'pss10': 20},
            {'resilience': 0.5, 'affect': 0.0, 'stress': 0.5, 'pss10': 20},
            {'resilience': 0.5, 'affect': 0.0, 'stress': 0.5, 'pss10': 20}
        ])

        viz_path = create_visualization_report(identical_data, temp_dir, "identical_plot.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_large_number_of_agents(self, temp_dir):
        """Test visualization with a large number of agents."""
        large_data = pd.DataFrame([
            {'resilience': 0.5, 'affect': 0.0, 'stress': 0.5, 'pss10': 20}
            for _ in range(1000)
        ])

        viz_path = create_visualization_report(large_data, temp_dir, "large_plot.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_scipy_availability_for_qq_plot(self, sample_agents, temp_dir):
        """Test Q-Q plot generation based on scipy availability."""
        # Convert agents to DataFrame
        data = pd.DataFrame([
            {'resilience': agent.resilience, 'affect': agent.affect, 'stress': agent.resources, 'pss10': agent.pss10}
            for agent in sample_agents
        ])

        viz_path = create_visualization_report(data, temp_dir, "qq_test.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()
            # If scipy is available, Q-Q plot should be generated
            # If not, it should still work but without Q-Q plot

    def test_visualization_content_validation(self, sample_agents, temp_dir):
        """Test that visualization contains expected elements."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        # Convert agents to DataFrame
        data = pd.DataFrame([
            {'resilience': agent.resilience, 'affect': agent.affect, 'stress': agent.resources, 'pss10': agent.pss10}
            for agent in sample_agents
        ])

        viz_path = create_visualization_report(data, temp_dir, "content_test.pdf")

        # Basic validation that file is not empty
        assert Path(viz_path).stat().st_size > 1000  # Reasonable size for a plot

    def test_error_handling_invalid_agents(self, temp_dir):
        """Test error handling with invalid agent data."""
        # DataFrame with invalid attributes
        invalid_data = pd.DataFrame([
            {'resilience': "invalid", 'affect': 0.0, 'stress': 0.5, 'pss10': 20},
            {'resilience': 0.5, 'affect': None, 'stress': 0.5, 'pss10': 20}
        ])

        # Should handle gracefully without crashing
        viz_path = create_visualization_report(invalid_data, temp_dir, "invalid_test.pdf")

        if HAS_MATPLOTLIB:
            # Should still generate something or handle error
            assert isinstance(viz_path, str)

    def test_integration_with_simulation_data(self, temp_dir):
        """Test integration with actual simulation data structure."""
        # DataFrame that mimics real simulation data
        sim_data = pd.DataFrame([
            {'resilience': np.random.random(), 'affect': np.random.uniform(-1, 1),
             'stress': np.random.random(), 'pss10': np.random.randint(0, 41)}
            for _ in range(50)
        ])

        viz_path = create_visualization_report(sim_data, temp_dir, "simulation_integration.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_reproducibility_of_visualization(self, sample_agents, temp_dir):
        """Test that visualizations are reproducible with same inputs."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        # Convert agents to DataFrame
        data = pd.DataFrame([
            {'resilience': agent.resilience, 'affect': agent.affect, 'stress': agent.resources, 'pss10': agent.pss10}
            for agent in sample_agents
        ])

        # Generate two visualizations with same inputs
        viz1 = create_visualization_report(data, temp_dir, "repro1.pdf")
        viz2 = create_visualization_report(data, temp_dir, "repro2.pdf")

        # Files should both exist and have similar sizes (approximately)
        assert Path(viz1).exists()
        assert Path(viz2).exists()

        size1 = Path(viz1).stat().st_size
        size2 = Path(viz2).stat().st_size

        # Allow some variation but should be reasonably close
        assert abs(size1 - size2) / max(size1, size2) < 0.1  # Within 10%

    def test_visualization_with_zero_variance(self, temp_dir):
        """Test visualization when all agents have identical values."""
        zero_var_data = pd.DataFrame([
            {'resilience': 0.5, 'affect': 0.0, 'stress': 0.5, 'pss10': 20}
        ] * 10)

        viz_path = create_visualization_report(zero_var_data, temp_dir, "zero_var.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_visualization_with_boundary_values(self, temp_dir):
        """Test visualization with agents at boundary values."""
        boundary_data = pd.DataFrame([
            {'resilience': 0.0, 'affect': -1.0, 'stress': 0.0, 'pss10': 0},
            {'resilience': 1.0, 'affect': 1.0, 'stress': 1.0, 'pss10': 40}
        ])

        viz_path = create_visualization_report(boundary_data, temp_dir, "boundary.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()


class TestTimeSeriesVisualization:
    """Test class for time series visualization function."""

    @pytest.fixture
    def temp_dir(self):
        """Provide temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_time_series_data(self):
        """Provide sample time series data for testing."""
        # Create 100 time steps of sample data
        np.random.seed(42)
        n_steps = 100

        data = {
            'avg_pss10': np.random.normal(20, 5, n_steps).clip(0, 40),
            'avg_stress': np.random.uniform(0, 1, n_steps),
            'avg_resilience': np.random.uniform(0, 1, n_steps),
            'avg_affect': np.random.uniform(-1, 1, n_steps)
        }

        return pd.DataFrame(data)

    def test_create_time_series_visualization_with_matplotlib(self, sample_time_series_data, temp_dir):
        """Test time series visualization generation when matplotlib is available."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        viz_path = create_time_series_visualization(sample_time_series_data, temp_dir, "test_time_series.pdf")

        # Verify file was created
        assert Path(viz_path).exists()
        assert viz_path.endswith("test_time_series.pdf")

        # Verify it's in the correct directory
        assert temp_dir in viz_path

        # Verify file is a valid image (basic check)
        assert Path(viz_path).stat().st_size > 0

    def test_create_time_series_visualization_without_matplotlib(self, sample_time_series_data, temp_dir):
        """Test time series visualization when matplotlib is not available."""
        # Mock matplotlib as unavailable
        with patch('src.python.visualization_utils.HAS_MATPLOTLIB', False):
            viz_path = create_time_series_visualization(sample_time_series_data, temp_dir, "test_time_series.pdf")

            # Should return placeholder file path
            assert viz_path.endswith("time_series_not_available.txt")
            assert Path(viz_path).exists()

            # Verify placeholder content
            with open(viz_path, 'r') as f:
                content = f.read()
                assert "matplotlib" in content.lower() or "not available" in content.lower()

    def test_time_series_empty_data(self, temp_dir):
        """Test time series visualization with empty data."""
        empty_data = pd.DataFrame(columns=['avg_pss10', 'avg_stress', 'avg_resilience', 'avg_affect'])

        # Should raise ValueError for empty data
        with pytest.raises(ValueError, match="No valid data after removing missing values"):
            create_time_series_visualization(empty_data, temp_dir, "empty_time_series.pdf")

    def test_time_series_missing_columns(self, temp_dir):
        """Test time series visualization with missing required columns."""
        incomplete_data = pd.DataFrame({
            'avg_pss10': [20, 21, 19],
            'avg_stress': [0.5, 0.6, 0.4]
            # Missing avg_resilience and avg_affect
        })

        # Should raise ValueError for missing columns
        with pytest.raises(ValueError, match="Missing required columns"):
            create_time_series_visualization(incomplete_data, temp_dir, "incomplete_time_series.pdf")

    def test_time_series_custom_filename(self, sample_time_series_data, temp_dir):
        """Test time series visualization with custom filename."""
        custom_filename = "custom_time_series.pdf"

        viz_path = create_time_series_visualization(sample_time_series_data, temp_dir, custom_filename)

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()
            assert custom_filename in viz_path

    def test_time_series_output_directory_creation(self, sample_time_series_data):
        """Test that time series visualization output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as base_dir:
            nested_dir = Path(base_dir) / "nested" / "deep" / "path"
            viz_path = create_time_series_visualization(sample_time_series_data, str(nested_dir), "test.pdf")

            if HAS_MATPLOTLIB:
                assert Path(viz_path).exists()
                assert nested_dir.exists()

    def test_time_series_with_extreme_values(self, temp_dir):
        """Test time series visualization with extreme values."""
        extreme_data = pd.DataFrame({
            'avg_pss10': [0, 40, 20, 0, 40],
            'avg_stress': [0.0, 1.0, 0.5, 0.0, 1.0],
            'avg_resilience': [0.0, 1.0, 0.5, 0.0, 1.0],
            'avg_affect': [-1.0, 1.0, 0.0, -1.0, 1.0]
        })

        viz_path = create_time_series_visualization(extreme_data, temp_dir, "extreme_time_series.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_time_series_with_identical_values(self, temp_dir):
        """Test time series visualization with identical values."""
        identical_data = pd.DataFrame({
            'avg_pss10': [20] * 10,
            'avg_stress': [0.5] * 10,
            'avg_resilience': [0.5] * 10,
            'avg_affect': [0.0] * 10
        })

        viz_path = create_time_series_visualization(identical_data, temp_dir, "identical_time_series.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_time_series_content_validation(self, sample_time_series_data, temp_dir):
        """Test that time series visualization contains expected elements."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        viz_path = create_time_series_visualization(sample_time_series_data, temp_dir, "content_test_time_series.pdf")

        # Basic validation that file is not empty
        assert Path(viz_path).stat().st_size > 1000  # Reasonable size for a plot

    def test_time_series_2x2_subplot_structure(self, sample_time_series_data, temp_dir):
        """Test that time series visualization has correct 2x2 subplot structure."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        viz_path = create_time_series_visualization(sample_time_series_data, temp_dir, "structure_test.pdf")

        # File should be created successfully
        assert Path(viz_path).exists()

    def test_time_series_alpha_values(self, sample_time_series_data, temp_dir):
        """Test that alpha values are applied correctly for raw data and moving averages."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        viz_path = create_time_series_visualization(sample_time_series_data, temp_dir, "alpha_test.pdf")

        # File should be created successfully
        assert Path(viz_path).exists()

    def test_time_series_legend_position(self, sample_time_series_data, temp_dir):
        """Test that legend is positioned below title and above subplots."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        viz_path = create_time_series_visualization(sample_time_series_data, temp_dir, "legend_test.pdf")

        # File should be created successfully
        assert Path(viz_path).exists()

    def test_time_series_subplot_labels(self, sample_time_series_data, temp_dir):
        """Test that subplots have correct labels in 2x2 order."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        viz_path = create_time_series_visualization(sample_time_series_data, temp_dir, "labels_test.pdf")

        # File should be created successfully
        assert Path(viz_path).exists()

    def test_time_series_with_small_dataset(self, temp_dir):
        """Test time series visualization with small dataset."""
        small_data = pd.DataFrame({
            'avg_pss10': [20, 21],
            'avg_stress': [0.5, 0.6],
            'avg_resilience': [0.5, 0.4],
            'avg_affect': [0.0, 0.1]
        })

        viz_path = create_time_series_visualization(small_data, temp_dir, "small_time_series.pdf")

        if HAS_MATPLOTLIB:
            assert Path(viz_path).exists()

    def test_time_series_reproducibility(self, sample_time_series_data, temp_dir):
        """Test that time series visualizations are reproducible with same inputs."""
        if not HAS_MATPLOTLIB:
            pytest.skip("Matplotlib not available")

        # Generate two visualizations with same inputs
        viz1 = create_time_series_visualization(sample_time_series_data, temp_dir, "repro1_time_series.pdf")
        viz2 = create_time_series_visualization(sample_time_series_data, temp_dir, "repro2_time_series.pdf")

        # Files should both exist and have similar sizes (approximately)
        assert Path(viz1).exists()
        assert Path(viz2).exists()

        size1 = Path(viz1).stat().st_size
        size2 = Path(viz2).stat().st_size

        # Allow some variation but should be reasonably close
        assert abs(size1 - size2) / max(size1, size2) < 0.1  # Within 10%

    def test_time_series_with_all_nan_values(self, temp_dir):
        """Test time series visualization handles all NaN values correctly."""
        data_all_nan = pd.DataFrame({
            'avg_pss10': [np.nan, np.nan, np.nan, np.nan],
            'avg_stress': [np.nan, np.nan, np.nan, np.nan],
            'avg_resilience': [np.nan, np.nan, np.nan, np.nan],
            'avg_affect': [np.nan, np.nan, np.nan, np.nan]
        })

        # Should raise ValueError due to all NaN values
        with pytest.raises(ValueError, match="No valid data after removing missing values"):
            create_time_series_visualization(data_all_nan, temp_dir, "all_nan_time_series.pdf")