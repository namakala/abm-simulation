"""Tests for Fix 2: Reduce stress decay rate and add stress floor.

Verifies:
- ASSUMPTION_STRESS_DECAY_RATE default is 0.08 (was 0.20)
- compute_stress_decay applies a minimum floor of 0.03
"""

import pytest

from src.python.assumption_config import AssumptionStressConfig
from src.python.affect_utils import compute_stress_decay, StressProcessingConfig


class TestStressDecayRate:
    """Fix 2a: Stress decay rate reduced to 0.08."""

    @pytest.mark.unit
    def test_stress_decay_rate_default_is_0_08(self):
        """ASSUMPTION_STRESS_DECAY_RATE default is 0.08 (was 0.20)."""
        c = AssumptionStressConfig()
        assert c.stress_decay_rate == 0.08, f"Expected 0.08, got {c.stress_decay_rate}"

    @pytest.mark.unit
    def test_stress_decays_at_8_percent(self):
        """compute_stress_decay with decay_rate=0.08 reduces stress by 8%."""
        config = StressProcessingConfig(stress_decay_rate=0.08)
        stress = 0.5
        decayed = compute_stress_decay(stress, config)
        expected = 0.5 * (1.0 - 0.08)  # 0.46
        assert abs(decayed - expected) < 1e-10, f"Expected {expected:.4f}, got {decayed:.4f}"


class TestStressFloor:
    """Fix 2b: Stress floor prevents asymptote to absolute zero."""

    @pytest.mark.unit
    def test_stress_floor_applied_when_below_floor(self):
        """compute_stress_decay never returns below 0.03."""
        config = StressProcessingConfig(stress_decay_rate=0.20)

        # Very low stress should not decay below floor
        decayed = compute_stress_decay(0.03, config)
        assert decayed >= 0.03, f"Expected ≥0.03, got {decayed:.6f}"

        # Stress at floor should stay at floor
        decayed = compute_stress_decay(0.031, config)  # 0.031 * 0.8 = 0.0248 < 0.03
        assert decayed >= 0.03, f"Expected ≥0.03, got {decayed:.6f}"

    @pytest.mark.unit
    def test_stress_above_floor_decays_normally(self):
        """Stress well above floor decays without floor intervention."""
        config = StressProcessingConfig(stress_decay_rate=0.08)
        decayed = compute_stress_decay(0.5, config)
        expected = 0.5 * (1.0 - 0.08)
        assert abs(decayed - expected) < 1e-10, f"Expected {expected:.4f}, got {decayed:.4f}"

    @pytest.mark.unit
    def test_stress_floor_with_high_decay_rate(self):
        """Floor kicks in reliably even with higher decay rates."""
        config = StressProcessingConfig(stress_decay_rate=0.50)
        decayed = compute_stress_decay(0.05, config)  # 0.05 * 0.5 = 0.025 < 0.03
        assert decayed >= 0.03, f"Expected ≥0.03, got {decayed:.6f}"
