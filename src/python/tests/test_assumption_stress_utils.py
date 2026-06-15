"""
Tests that stress_utils functions use ASSUMPTION_* config.

Verifies env var override propagation into the core utility function
defaults for stress dimension updates, event difficulty, etc.

NOTE: Avoids importlib.reload on modules with dataclasses (PSS10Item)
to prevent isinstance breakage. Instead, tests validate that the
assumption config values are correctly referenced by the functions.
"""

import os
import pytest


class TestComputeEventDifficultyAssumptions:
    """compute_event_difficulty uses assumption config weights."""

    @pytest.mark.unit
    def test_default_challenge_weight(self):
        """Default challenge weight is 0.7."""
        from src.python.stress_utils import compute_event_difficulty

        # challenge=1.0, hindrance=0.0: 1.0*0.7 + 0.0*1.3 = 0.7
        diff = compute_event_difficulty(1.0, 0.0)
        assert diff == pytest.approx(0.7, abs=0.01)

    @pytest.mark.unit
    def test_default_hindrance_weight(self):
        """Default hindrance weight is 1.3."""
        from src.python.stress_utils import compute_event_difficulty

        # challenge=0.0, hindrance=1.0: 0.0*0.7 + 1.0*1.3 = 1.3
        diff = compute_event_difficulty(0.0, 1.0)
        assert diff == pytest.approx(1.3, abs=0.01)

    @pytest.mark.unit
    def test_env_override_challenge_weight(self, clean_env):
        """ASSUMPTION_EVENT_INTENSITY_CHALLENGE_WEIGHT changes result."""
        from src.python.stress_utils import compute_event_difficulty
        from src.python.assumption_config import reload_assumptions

        os.environ["ASSUMPTION_EVENT_INTENSITY_CHALLENGE_WEIGHT"] = "0.5"
        try:
            reload_assumptions()
            diff = compute_event_difficulty(1.0, 0.0)
            assert diff == pytest.approx(0.5, abs=0.01)
        finally:
            os.environ.pop("ASSUMPTION_EVENT_INTENSITY_CHALLENGE_WEIGHT", None)
            reload_assumptions()


class TestUpdateStressDimensionsAssumptions:
    """update_stress_dimensions_from_event uses assumption config defaults."""

    @pytest.mark.unit
    def test_controllability_challenge_default(self):
        """Default controllability_challenge_weight is 0.10."""
        from src.python.stress_utils import update_stress_dimensions_from_event

        ctrl, overload, intensity, momentum = update_stress_dimensions_from_event(
            current_controllability=0.5,
            current_overload=0.5,
            challenge=1.0,
            hindrance=0.0,
            coped_successfully=True,
            is_stressful=True,
            volatility=1.0,
            resilience=0.0,  # disable buffer for assumption testing
        )
        # controllability_change_magnitude = 1.0*0.10 + 0.0*0.05 = 0.10
        # homeostasis = (0.5 - 0.5) * 0.05 = 0
        # event = 0.10 * 1.0 * resilience_buffer(1.0) = 0.10
        # expected: 0.5 + 0 + 0.10 = 0.60
        assert ctrl == pytest.approx(0.60, abs=0.01)

    @pytest.mark.unit
    def test_env_override_controllability_challenge(self, clean_env):
        """ASSUMPTION_CONTROLLABILITY_CHALLENGE_WEIGHT changes behavior."""
        from src.python.stress_utils import update_stress_dimensions_from_event
        from src.python.assumption_config import reload_assumptions

        os.environ["ASSUMPTION_CONTROLLABILITY_CHALLENGE_WEIGHT"] = "0.20"
        try:
            reload_assumptions()
            ctrl, overload, intensity, momentum = update_stress_dimensions_from_event(
                current_controllability=0.5,
                current_overload=0.5,
                challenge=1.0,
                hindrance=0.0,
                coped_successfully=True,
                is_stressful=True,
                volatility=1.0,
                resilience=0.0,  # disable buffer for assumption testing
            )
            # With weight=0.20: ctrl = 0.5 + (1.0*0.20) * 1.0 * 1.0 = 0.70
            assert ctrl == pytest.approx(0.70, abs=0.01)
        finally:
            os.environ.pop("ASSUMPTION_CONTROLLABILITY_CHALLENGE_WEIGHT", None)
            reload_assumptions()

    @pytest.mark.unit
    def test_env_override_homeostasis_all_stressful(self, clean_env):
        """Homeostasis rate affects controllability for stressful events."""
        from src.python.stress_utils import update_stress_dimensions_from_event
        from src.python.assumption_config import reload_assumptions

        os.environ["ASSUMPTION_CONTROLLABILITY_HOMEOSTASIS_RATE"] = "0.10"
        try:
            reload_assumptions()
            ctrl, overload, intensity, momentum = update_stress_dimensions_from_event(
                current_controllability=0.8,  # Above baseline
                current_overload=0.5,
                challenge=0.0,
                hindrance=0.0,
                coped_successfully=True,
                is_stressful=True,
                volatility=1.0,
            )
            # homeostasis = (0.5 - 0.8) * 0.10 = -0.03
            # event_effect = 0.0 * 1.0 = 0
            # expected: 0.8 - 0.03 = 0.77
            assert ctrl == pytest.approx(0.77, abs=0.01)
        finally:
            os.environ.pop("ASSUMPTION_CONTROLLABILITY_HOMEOSTASIS_RATE", None)
            reload_assumptions()


class TestEstimatePSS10Assumptions:
    """estimate_pss10_from_stress_dimensions uses assumption config."""

    @pytest.mark.unit
    def test_default_estimation(self):
        """Default PSS-10 estimation yields correct range."""
        from src.python.stress_utils import estimate_pss10_from_stress_dimensions

        low, high = estimate_pss10_from_stress_dimensions(0.5, 0.5)
        # base=10, ctl_effect=0.5*8=4, ovr_effect=0.5*12=6
        # estimated = 10+4+6 = 20, variance=3 => (17, 23)
        assert low <= 20 <= high

    @pytest.mark.unit
    def test_env_override_base(self, clean_env):
        """ASSUMPTION_PSS10_ESTIMATION_BASE changes estimation."""
        from src.python.stress_utils import estimate_pss10_from_stress_dimensions
        from src.python.assumption_config import reload_assumptions

        os.environ["ASSUMPTION_PSS10_ESTIMATION_BASE"] = "15"
        try:
            reload_assumptions()
            low, high = estimate_pss10_from_stress_dimensions(0.5, 0.5)
            # base=15, ctl_effect=4, ovr_effect=6 => 25, variance=3 => (22, 28)
            assert low <= 25 <= high
        finally:
            os.environ.pop("ASSUMPTION_PSS10_ESTIMATION_BASE", None)
            reload_assumptions()


class TestDecayStressIntensityAssumptions:
    """decay_recent_stress_intensity uses assumption config."""

    @pytest.mark.unit
    def test_default_momentum_threshold(self):
        """Default momentum_zero_threshold is 0.01."""
        from src.python.stress_utils import decay_recent_stress_intensity

        _, momentum = decay_recent_stress_intensity(0.5, 0.5)
        # momentum > 0.01 so: momentum * 0.9 = 0.45
        assert momentum == pytest.approx(0.45, abs=0.01)
