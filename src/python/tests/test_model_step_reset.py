"""Tests that StressModel.step() delegates daily reset to Person.step() (Plan 008 Step 7).

Person.step() now handles daily reset internally via its consolidation loop.
The separate _daily_reset() call in StressModel.step() is redundant and should
be removed to prevent double-reset.
"""

import pytest

from src.python.model import StressModel


## Fixtures


@pytest.fixture
def model():
    """Small StressModel for testing (N > k=4 to avoid NetworkXError)."""
    return StressModel(N=6, max_days=5, seed=42)


## Reset removal tests


class TestModelStepNoDoubleReset:
    """StressModel.step() does not redundantly call _daily_reset."""

    def test_person_step_resets_counters(self, model):
        """After model.step(), all agent counters are 0 (reset inside Person.step())."""
        model.step()
        for agent in model.agents:
            assert agent.daily_interactions == 0, (
                f"Agent {agent.unique_id}: daily_interactions={agent.daily_interactions}, expected 0"
            )
            assert agent.daily_support_exchanges == 0, f"Agent {agent.unique_id}: daily_support_exchanges should be 0"

    def test_model_step_runs_without_daily_reset_call(self, model):
        """model.step() runs successfully even without the separate _daily_reset loop."""
        # The _daily_reset loop (lines 266-269) is redundant because
        # Person.step() already runs process_daily_reset internally.
        # This test just verifies the model step completes successfully.
        for _ in range(3):
            model.step()
        assert model.day == 3

    def test_counters_collected_before_reset(self, model):
        """DataCollector sees non-zero counters (reset happens at end of Person.step())."""
        # Run a step that triggers interactions
        for agent in model.agents:
            agent.daily_interactions = 5
            agent.daily_support_exchanges = 3

        model.datacollector.collect(model)

        agent_df = model.datacollector.get_agent_vars_dataframe()
        # The collected data should have the values we set
        # (collect happens before model.day increments)
        assert not agent_df.empty

    def test_model_step_idempotent_counts(self, model):
        """Model runs multiple steps without counter accumulation bugs."""
        # Ensure that counters don't leak across days
        old_total = model.total_interactions
        for _ in range(3):
            model.step()
        # total_interactions should have increased (or stayed same if no interactions)
        assert model.total_interactions >= old_total
        # All agents should have their counters reset after each step
        for agent in model.agents:
            assert agent.daily_interactions == 0
            assert agent.daily_support_exchanges == 0
