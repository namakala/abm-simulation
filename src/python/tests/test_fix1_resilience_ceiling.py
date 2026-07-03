"""Tests for Fix 1: Ceiling damping on resilience gains.

Verifies that resilience delta is scaled by (1 - R) so that
growth slows as R approaches 1.0, preventing ceiling saturation.
"""

import pytest

from src.python.affect_utils import compute_challenge_hindrance_resilience_effect


class TestResilienceCeilingDamping:
    """Fix 1: Diminishing returns on resilience gains near ceiling."""

    @pytest.mark.unit
    def test_damping_at_low_resilience(self):
        """At R=0.3, damping factor is ~0.7, so delta is still substantial."""
        effect = compute_challenge_hindrance_resilience_effect(
            challenge=0.9, hindrance=0.1, coped_successfully=True, current_resilience=0.3
        )
        # Without damping: +0.3 * 0.9 + 0.1 * 0.1 = 0.27 + 0.01 = 0.28
        # With damping (1 - 0.3) = 0.7: 0.28 * 0.7 = 0.196
        assert 0.10 < effect < 0.28, f"Expected moderate damping at R=0.3, got {effect:.4f}"

    @pytest.mark.unit
    def test_strong_damping_at_high_resilience(self):
        """At R=0.9, damping factor is ~0.1, delta should be much smaller."""
        effect = compute_challenge_hindrance_resilience_effect(
            challenge=0.9, hindrance=0.1, coped_successfully=True, current_resilience=0.9
        )
        # Without damping: +0.28, with (1 - 0.9) = 0.1: 0.028
        assert effect < 0.05, f"Expected strong damping near ceiling, got {effect:.4f}"

    @pytest.mark.unit
    def test_damping_scales_with_resilience_level(self):
        """Higher R produces smaller delta for identical event."""
        effect_low = compute_challenge_hindrance_resilience_effect(
            challenge=0.8, hindrance=0.2, coped_successfully=True, current_resilience=0.3
        )
        effect_high = compute_challenge_hindrance_resilience_effect(
            challenge=0.8, hindrance=0.2, coped_successfully=True, current_resilience=0.8
        )
        assert effect_low > effect_high, (
            f"Lower R ({effect_low:.4f}) should give larger delta than higher R ({effect_high:.4f})"
        )

    @pytest.mark.unit
    def test_damping_on_failure_too(self):
        """Failed coping also gets damped near ceiling (negative delta shrinks)."""
        effect_low = compute_challenge_hindrance_resilience_effect(
            challenge=0.1, hindrance=0.9, coped_successfully=False, current_resilience=0.3
        )
        effect_high = compute_challenge_hindrance_resilience_effect(
            challenge=0.1, hindrance=0.9, coped_successfully=False, current_resilience=0.9
        )
        # Both negative, but high-resilience agent loses less (damped)
        assert effect_low < effect_high, f"High R ({effect_high:.4f}) should lose less than low R ({effect_low:.4f})"

    @pytest.mark.unit
    def test_zero_resilience_no_damping(self):
        """At R=0, damping factor = 1.0 — no damping."""
        effect = compute_challenge_hindrance_resilience_effect(
            challenge=0.9, hindrance=0.1, coped_successfully=True, current_resilience=0.0
        )
        expected_undamped = 0.3 * 0.9 + 0.1 * 0.1  # 0.28
        assert abs(effect - expected_undamped) < 1e-10, (
            f"Expected no damping at R=0: {expected_undamped}, got {effect:.4f}"
        )

    @pytest.mark.unit
    def test_near_ceiling_resilience_almost_no_change(self):
        """At R very close to 1.0, delta approaches zero."""
        effect = compute_challenge_hindrance_resilience_effect(
            challenge=1.0, hindrance=0.0, coped_successfully=True, current_resilience=0.99
        )
        assert abs(effect) < 0.01, f"Expected near-zero effect at R=0.99, got {effect:.6f}"

    @pytest.mark.unit
    def test_default_resilience_preserves_current_behavior(self):
        """Calling without current_resilience defaults to R=0.5, giving 0.5x scaling."""
        effect = compute_challenge_hindrance_resilience_effect(challenge=0.9, hindrance=0.1, coped_successfully=True)
        # With default R=0.5: damping = 0.5, undamped = 0.28, damped = 0.14
        assert 0.10 < effect < 0.20, f"Expected ~0.14 with default R=0.5, got {effect:.4f}"
