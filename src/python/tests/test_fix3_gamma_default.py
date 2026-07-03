"""Tests for Fix 3: Reduce sigmoid gamma default from 6.0 to 3.0.

Verifies the config default changes and that the resulting
challenge/hindrance distribution is less bimodal.
"""

import pytest

from src.python.config import get_config


class TestGammaDefault:
    """Fix 3: Gamma default should be 3.0."""

    @pytest.mark.unit
    @pytest.mark.config
    def test_gamma_default_is_3_0(self):
        """APPRAISAL_GAMMA default is now 3.0 (was 6.0)."""
        cfg = get_config()
        gamma = cfg.get("appraisal", "gamma")
        assert gamma == 3.0, f"Expected gamma=3.0, got {gamma}"

    @pytest.mark.unit
    def test_gamma_env_var_override(self):
        """Setting APPRAISAL_GAMMA env var overrides the default."""
        import os

        os.environ["APPRAISAL_GAMMA"] = "1.5"
        # Reload config to pick up new env var
        from src.python.config import reload_config

        reload_config()
        try:
            cfg = get_config()
            assert cfg.get("appraisal", "gamma") == 1.5
        finally:
            del os.environ["APPRAISAL_GAMMA"]
            reload_config()

    @pytest.mark.unit
    def test_sigmoid_at_gamma_3_gives_graded_output(self):
        """At gamma=3.0, z=±0.5 gives challenge in (0.18, 0.82) — not near-binary."""
        from src.python.stress_utils import apply_weights
        from src.python.stress_utils import AppraisalWeights, StressEvent

        weights = AppraisalWeights(omega_c=1.0, omega_o=1.0, bias=0.0, gamma=3.0)

        # z = 0.5 (moderate controllability advantage)
        event_c = StressEvent(controllability=0.75, overload=0.25)
        challenge_c, hindrance_c = apply_weights(event_c, weights)
        assert 0.5 < challenge_c < 0.95, f"challenge should be graded, got {challenge_c:.4f}"
        assert 0.05 < hindrance_c < 0.5, f"hindrance should be graded, got {hindrance_c:.4f}"

        # z = -0.5 (moderate overload advantage)
        event_o = StressEvent(controllability=0.25, overload=0.75)
        challenge_o, hindrance_o = apply_weights(event_o, weights)
        assert 0.05 < challenge_o < 0.5, f"challenge should be graded, got {challenge_o:.4f}"
        assert 0.5 < hindrance_o < 0.95, f"hindrance should be graded, got {hindrance_o:.4f}"
