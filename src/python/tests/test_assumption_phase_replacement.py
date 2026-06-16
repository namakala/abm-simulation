"""
Tests that phase modules use ASSUMPTION_* config values instead of hardcoded constants.

Uses reload_assumptions() (safe, no class identity changes) and direct
env var tests on the assumption config values.
"""

import os
import pytest


class TestResilienceActivationAssumptions:
    """Resilience activation phase uses assumption config values."""

    @pytest.mark.unit
    def test_module_level_constants_match_assumptions(self):
        """Module-level constants equal assumption config defaults."""
        import importlib
        import src.python.phases.resilience_activation as ra
        from src.python.assumption_config import reload_assumptions, get_assumptions

        # Force fresh state — other tests may have polluted the cache
        reload_assumptions()
        importlib.reload(ra)

        a = get_assumptions()
        assert ra._RESOURCE_REWARD_MULTIPLIER == a.coping.resource_reward
        assert ra._RESOURCE_PENALTY_MULTIPLIER == a.coping.resource_penalty
        assert ra._PF_ALLOCATION_FRACTION == a.coping.pf_allocation_fraction

    @pytest.mark.unit
    def test_module_reload_picks_up_env(self, clean_env):
        """After env var set and reload_assumptions, new module load uses new value."""
        from src.python.assumption_config import reload_assumptions

        os.environ["ASSUMPTION_RESOURCE_REWARD"] = "0.5"
        try:
            reload_assumptions()
            # Re-import the phase module to trigger fresh module-level assignments
            import importlib
            import src.python.phases.resilience_activation as ra

            importlib.reload(ra)
            assert ra._RESOURCE_REWARD_MULTIPLIER == 0.5
        finally:
            os.environ.pop("ASSUMPTION_RESOURCE_REWARD", None)
            reload_assumptions()
            # Restore defaults by re-importing
            import importlib
            import src.python.phases.resilience_activation as ra

            importlib.reload(ra)

    @pytest.mark.unit
    def test_env_override_pf_allocation_fraction(self, clean_env):
        """ASSUMPTION_PF_ALLOCATION_FRACTION changes module constant."""
        from src.python.assumption_config import reload_assumptions

        os.environ["ASSUMPTION_PF_ALLOCATION_FRACTION"] = "0.5"
        try:
            reload_assumptions()
            import importlib
            import src.python.phases.resilience_activation as ra

            importlib.reload(ra)
            assert ra._PF_ALLOCATION_FRACTION == 0.5
        finally:
            os.environ.pop("ASSUMPTION_PF_ALLOCATION_FRACTION", None)
            reload_assumptions()
            import importlib
            import src.python.phases.resilience_activation as ra

            importlib.reload(ra)

    @pytest.mark.unit
    def test_env_override_resource_penalty(self, clean_env):
        """ASSUMPTION_RESOURCE_PENALTY changes module constant."""
        from src.python.assumption_config import reload_assumptions

        os.environ["ASSUMPTION_RESOURCE_PENALTY"] = "0.2"
        try:
            reload_assumptions()
            import importlib
            import src.python.phases.resilience_activation as ra

            importlib.reload(ra)
            assert ra._RESOURCE_PENALTY_MULTIPLIER == 0.2
        finally:
            os.environ.pop("ASSUMPTION_RESOURCE_PENALTY", None)
            reload_assumptions()
            import importlib
            import src.python.phases.resilience_activation as ra

            importlib.reload(ra)


class TestResourceAllocationAssumptions:
    """Resource allocation phase uses assumption config values."""

    @pytest.mark.unit
    def test_module_level_constants_match_assumptions(self):
        """Module-level constants equal assumption config defaults.

        Uses importlib.reload to force re-evaluation of module-level constants.
        """
        import importlib

        # Force-reload so module-level consts pick up current assumptions
        from src.python import phases as _phases

        importlib.reload(_phases.resource_allocation)

        from src.python.phases.resource_allocation import (
            _AFFECT_MULT_COEFFICIENT,
            _RESILIENCE_MULT_COEFFICIENT,
            _RESILIENCE_BONUS_FACTOR,
            _EFFICIENCY_RETURN_FACTOR,
        )
        from src.python.assumption_config import get_assumptions

        a = get_assumptions()
        assert _AFFECT_MULT_COEFFICIENT == a.resource.affect_regeneration_multiplier
        assert _RESILIENCE_MULT_COEFFICIENT == a.resource.resilience_regeneration_multiplier
        assert _RESILIENCE_BONUS_FACTOR == a.resource.challenge_resilience_bonus_factor
        assert _EFFICIENCY_RETURN_FACTOR == a.resource.efficiency_return_factor


class TestStressBufferingAssumptions:
    """Stress buffering phase uses assumption config values."""

    @pytest.mark.unit
    def test_module_level_constants_match_assumptions(self):
        """Module-level defaults match assumption config."""
        from src.python.phases.stress_buffering import (
            _DEFAULT_BOOST_RATE,
            _DEFAULT_A_COEFFICIENT,
            _DEFAULT_B_COEFFICIENT,
            _DEFAULT_C_PRIME_COEFFICIENT,
        )
        from src.python.assumption_config import get_assumptions

        a = get_assumptions()
        assert _DEFAULT_BOOST_RATE == a.resource.buffering_boost_rate
        assert _DEFAULT_A_COEFFICIENT == a.resource.buffering_a_coefficient
        assert _DEFAULT_B_COEFFICIENT == a.resource.buffering_b_coefficient
        assert _DEFAULT_C_PRIME_COEFFICIENT == a.resource.buffering_c_prime_coefficient
