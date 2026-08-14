"""Tests for Person._last_phase_outputs instrumentation.

Verifies that after a model step, each agent's _last_phase_outputs dict
contains the expected PhaseOutput keys from all phases.
"""

from __future__ import annotations

from src.python.model import StressModel


class TestLastPhaseOutputs:
    """Person.step() stores phase outputs for demo metric extraction."""

    def test_dict_exists_after_step(self):
        """_last_phase_outputs is a dict after model.step()."""
        # N must be > network degree param k (default=4) to avoid k>n error
        model = StressModel(N=5, max_days=3, seed=42)
        model.step()
        for agent in model.agents:
            assert isinstance(agent._last_phase_outputs, dict)

    def test_contains_expected_keys(self):
        """_last_phase_outputs has keys for all phase types after a step."""
        model = StressModel(N=5, max_days=2, seed=42)
        model.step()
        agent = list(model.agents)[0]
        keys = agent._last_phase_outputs.keys()
        # Must have daily consolidation phases
        assert "stress_perception" in keys or "affect_dynamics" in keys
        assert "resource_allocation" in keys or "stress_buffering" in keys
        assert "daily_reset" in keys
        assert "pss10_consolidation" in keys

    def test_values_are_phase_output_dicts(self):
        """Each _last_phase_outputs value has state_delta and observation keys."""
        model = StressModel(N=5, max_days=3, seed=42)
        model.step()
        agent = list(model.agents)[0]
        for phase_name, output in agent._last_phase_outputs.items():
            assert "state_delta" in output, f"{phase_name} missing state_delta"
            assert "observation" in output, f"{phase_name} missing observation"
            assert isinstance(output["state_delta"], dict)
            assert isinstance(output["observation"], dict)

    def test_reset_each_step(self):
        """_last_phase_outputs is reset each step (starts empty)."""
        model = StressModel(N=5, max_days=3, seed=42)
        # Step once
        model.step()
        agent = list(model.agents)[0]
        assert len(agent._last_phase_outputs) > 0

        # Step again — old outputs should be replaced
        previous = dict(agent._last_phase_outputs)
        model.step()
        # Some keys should still be there, but values may differ
        for key in previous:
            if key in agent._last_phase_outputs:
                assert agent._last_phase_outputs[key] is not previous[key], f"{key} was not refreshed"

    def test_buffering_and_allocation_present(self):
        """Stress buffering and resource allocation outputs are captured."""
        model = StressModel(N=5, max_days=3, seed=42)
        model.step()
        agent = list(model.agents)[0]
        keys = agent._last_phase_outputs
        # At minimum, daily consolidation phases should be present
        assert "resource_allocation" in keys, f"Missing resource_allocation in {list(keys.keys())}"
        assert "stress_buffering" in keys, f"Missing stress_buffering in {list(keys.keys())}"

    def test_daily_cycle_keys_present(self):
        """Daily cycle phase outputs are captured."""
        model = StressModel(N=5, max_days=3, seed=42)
        model.step()
        agent = list(model.agents)[0]
        keys = agent._last_phase_outputs
        assert "affect_dynamics" in keys, f"Missing affect_dynamics in {list(keys.keys())}"
        assert "daily_reset" in keys, f"Missing daily_reset in {list(keys.keys())}"
        assert "pss10_consolidation" in keys, f"Missing pss10_consolidation in {list(keys.keys())}"
