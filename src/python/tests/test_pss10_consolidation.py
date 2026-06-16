"""
Tests for the PSS-10 partial daily-event stress moving average.

Verifies the asymmetric consolidation algorithm:
- Higher-than-average stress events are recorded directly
- Lower-than-average stress events are blended with the running average
- End-of-day consolidation produces a rounded mean
"""

import pytest


class TestAppendDailyPss10Score:
    """append_daily_pss10_score implements the partial moving average."""

    @pytest.mark.unit
    def test_append_first_score(self):
        """First score is always appended directly."""
        from src.python.stress_utils import append_daily_pss10_score

        scores = []
        result = append_daily_pss10_score(scores, 3)
        assert result == [3]

    @pytest.mark.unit
    def test_append_higher_score_direct(self):
        """Score higher than previous average is appended directly."""
        from src.python.stress_utils import append_daily_pss10_score

        scores = [3]
        result = append_daily_pss10_score(scores, 9)
        # avg([3]) = 3, 9 > 3 → direct append
        assert result == [3, 9]

    @pytest.mark.unit
    def test_append_lower_score_blended(self):
        """Score lower than previous average is blended."""
        from src.python.stress_utils import append_daily_pss10_score

        scores = [3, 9]
        result = append_daily_pss10_score(scores, 4)
        # avg([3, 9]) = 6, 4 < 6 → blend: round((4+6)/2) = 5
        assert result == [3, 9, 5]

    @pytest.mark.unit
    def test_complex_sequence(self):
        """Full example from specification: [3, 9, 4, 5] → [3, 9, 5, 5]."""
        from src.python.stress_utils import append_daily_pss10_score

        scores = []
        scores = append_daily_pss10_score(scores, 3)  # [3]
        scores = append_daily_pss10_score(scores, 9)  # [3, 9]
        scores = append_daily_pss10_score(scores, 4)  # [3, 9, 5]
        scores = append_daily_pss10_score(scores, 5)  # [3, 9, 5, 5]
        assert scores == [3, 9, 5, 5]

    @pytest.mark.unit
    def test_persistent_low_scores_decay(self):
        """Repeated low scores gradually pull average down."""
        from src.python.stress_utils import append_daily_pss10_score

        scores = []
        scores = append_daily_pss10_score(scores, 9)  # [9]
        scores = append_daily_pss10_score(scores, 3)  # avg(9)=9, 3<9 → blend: round((3+9)/2)=6
        scores = append_daily_pss10_score(scores, 3)  # avg(9,6)=7.5, 3<7.5 → blend: round((3+7.5)/2)=5
        scores = append_daily_pss10_score(scores, 3)  # avg(9,6,5)=6.67, 3<6.67 → blend: round((3+6.67)/2)=5
        assert scores == [9, 6, 5, 5]

    @pytest.mark.unit
    def test_steady_high_scores(self):
        """Consistently high scores are all recorded directly."""
        from src.python.stress_utils import append_daily_pss10_score

        scores = []
        for score in [7, 8, 9, 10]:
            scores = append_daily_pss10_score(scores, score)
        assert scores == [7, 8, 9, 10]

    @pytest.mark.unit
    def test_input_array_not_mutated(self):
        """Function returns a new list, does not mutate input."""
        from src.python.stress_utils import append_daily_pss10_score

        original = [3]
        result = append_daily_pss10_score(original, 9)
        assert original == [3]  # unchanged
        assert result == [3, 9]

    @pytest.mark.unit
    def test_high_then_low_then_high(self):
        """Pattern: high, low, high. High after blend is still recorded."""
        from src.python.stress_utils import append_daily_pss10_score

        scores = []
        scores = append_daily_pss10_score(scores, 8)  # [8]
        scores = append_daily_pss10_score(scores, 2)  # avg(8)=8, 2<8 → blend: round((2+8)/2)=5
        scores = append_daily_pss10_score(scores, 7)  # avg(8,5)=6.5, 7>6.5 → direct: [8,5,7]
        assert scores == [8, 5, 7]


class TestConsolidateDailyPss10:
    """consolidate_daily_pss10 computes final score from daily array."""

    @pytest.mark.unit
    def test_consolidate_empty_returns_none(self):
        """Empty array returns None (no stress events today)."""
        from src.python.stress_utils import consolidate_daily_pss10

        result = consolidate_daily_pss10([])
        assert result is None

    @pytest.mark.unit
    def test_consolidate_single_score(self):
        """Single score returns the score itself."""
        from src.python.stress_utils import consolidate_daily_pss10

        result = consolidate_daily_pss10([7])
        assert result == 7

    @pytest.mark.unit
    def test_consolidate_example(self):
        """Example from spec: [3, 9, 5, 5] → round(mean)=6."""
        from src.python.stress_utils import consolidate_daily_pss10

        result = consolidate_daily_pss10([3, 9, 5, 5])
        # mean = 5.5, round = 6
        assert result == 6

    @pytest.mark.unit
    def test_consolidate_multiple_scores(self):
        """Multiple scores produce rounded mean."""
        from src.python.stress_utils import consolidate_daily_pss10

        result = consolidate_daily_pss10([3, 9, 5])
        # mean = 5.67, round = 6
        assert result == 6


class TestSmoothPss10AcrossDays:
    """smooth_pss10_across_days applies exponential smoothing."""

    @pytest.mark.unit
    def test_smooth_first_day(self):
        """First day: alpha=0.30, prev=None returns consolidated as-is."""
        from src.python.stress_utils import smooth_pss10_across_days

        result = smooth_pss10_across_days(20, None, alpha=0.30)
        assert result == 20

    @pytest.mark.unit
    def test_smooth_second_day(self):
        """Second day: smoothed = alpha * consolidated + (1-alpha) * prev."""
        from src.python.stress_utils import smooth_pss10_across_days

        # prev=20, consolidated=14, alpha=0.30
        # smoothed = 0.30*14 + 0.70*20 = 4.2 + 14.0 = 18.2
        result = smooth_pss10_across_days(14, 20, alpha=0.30)
        assert result == pytest.approx(18.2, abs=0.01)

    @pytest.mark.unit
    def test_smooth_returns_float(self):
        """Returns float for downstream rounding, not int."""
        from src.python.stress_utils import smooth_pss10_across_days

        result = smooth_pss10_across_days(14, 20, alpha=0.30)
        assert isinstance(result, float)

    @pytest.mark.unit
    def test_smooth_custom_alpha(self):
        """Custom alpha changes the blend."""
        from src.python.stress_utils import smooth_pss10_across_days

        # alpha=0.10: 90% weight on prev, 10% on new
        result = smooth_pss10_across_days(14, 20, alpha=0.10)
        # 0.10*14 + 0.90*20 = 1.4 + 18.0 = 19.4
        assert result == pytest.approx(19.4, abs=0.01)

    @pytest.mark.unit
    def test_smooth_alpha_one(self):
        """alpha=1.0: only current day matters."""
        from src.python.stress_utils import smooth_pss10_across_days

        result = smooth_pss10_across_days(14, 20, alpha=1.0)
        assert result == 14.0
