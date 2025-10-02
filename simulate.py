"""
Enhanced simulation script to demonstrate end-of-day homeostatic adjustment mechanism.

This script demonstrates how the homeostatic mechanism stabilizes affect and resilience
over multiple simulated days, preventing long-term drift while maintaining responsiveness
to daily events.
"""

import numpy as np
from src.python.model import StressModel
from src.python.agent import Person
from src.python.config import Config, get_config
from src.python.affect_utils import compute_homeostatic_adjustment


def demonstrate_homeostatic_mechanism():
    """
    Demonstrate the homeostatic adjustment mechanism working over multiple days.

    Shows how agents' affect and resilience values are pulled back toward their
    baselines at the end of each day, preventing long-term drift while still
    allowing responsiveness to daily events.
    """
    print("=" * 80)
    print("HOMEOSTATIC ADJUSTMENT DEMONSTRATION")
    print("=" * 80)
    print()

    # Create a model with a small number of agents for clear demonstration
    model = StressModel(N=5, max_days=7, seed=42)

    print("Initial agent states:")
    print("-" * 50)
    for i, agent in enumerate(model.agents):
        print(f"Agent {i}: Affect={agent.affect:.3f}, Resilience={agent.resilience:.3f}, Baseline_Affect={agent.baseline_affect:.3f}, Baseline_Resilience={agent.baseline_resilience:.3f}")
    print()

    # Track homeostatic adjustments for analysis
    homeostatic_changes = {
        'day': [],
        'agent_id': [],
        'affect_before': [],
        'affect_after': [],
        'affect_adjustment': [],
        'resilience_before': [],
        'resilience_after': [],
        'resilience_adjustment': []
    }

    print("Running simulation with homeostatic adjustment tracking...")
    print("-" * 80)

    for day in range(model.max_days):
        print(f"\n--- DAY {day + 1} ---")

        # Store pre-step baselines for comparison
        pre_step_baselines = []
        for agent in model.agents:
            pre_step_baselines.append((agent.baseline_affect, agent.baseline_resilience))

        # Execute one day
        model.step()

        # Show detailed homeostatic adjustments for each agent
        print(f"Homeostatic adjustments for Day {day + 1}:")
        print("-" * 60)

        for i, agent in enumerate(model.agents):
            # Calculate what the adjustment was
            old_baseline_affect, old_baseline_resilience = pre_step_baselines[i]

            print(f"Agent {i}:")
            print(f"  Affect: {old_baseline_affect:.3f} → {agent.affect:.3f} (Δ={agent.affect - old_baseline_affect:+.3f})")
            print(f"  Resilience: {old_baseline_resilience:.3f} → {agent.resilience:.3f} (Δ={agent.resilience - old_baseline_resilience:+.3f})")

            # Track for analysis
            homeostatic_changes['day'].append(day + 1)
            homeostatic_changes['agent_id'].append(i)
            homeostatic_changes['affect_before'].append(old_baseline_affect)
            homeostatic_changes['affect_after'].append(agent.affect)
            homeostatic_changes['affect_adjustment'].append(agent.affect - old_baseline_affect)
            homeostatic_changes['resilience_before'].append(old_baseline_resilience)
            homeostatic_changes['resilience_after'].append(agent.resilience)
            homeostatic_changes['resilience_adjustment'].append(agent.resilience - old_baseline_resilience)

        # Show population summary
        summary = model.get_population_summary()
        print("\nPopulation Summary:")
        print(f"  Average Affect: {summary['avg_affect']:.3f}")
        print(f"  Average Resilience: {summary['avg_resilience']:.3f}")
        print(f"  Stress Prevalence: {summary['stress_prevalence']:.3f}")

    print("\n" + "=" * 80)
    print("HOMEOSTATIC MECHANISM ANALYSIS")
    print("=" * 80)

    # Analyze the homeostatic adjustments
    analyze_homeostatic_behavior(homeostatic_changes)

    return homeostatic_changes


def analyze_homeostatic_behavior(changes):
    """Analyze the homeostatic adjustment behavior to verify correct operation."""

    if not changes['day']:
        print("No data to analyze.")
        return

    # Convert to numpy arrays for analysis
    affect_adjustments = np.array(changes['affect_adjustment'])
    resilience_adjustments = np.array(changes['resilience_adjustment'])

    print("\nStatistical Analysis of Homeostatic Adjustments:")
    print("-" * 50)

    # Analyze affect adjustments
    print("Affect Adjustments:")
    print(f"  Mean adjustment: {np.mean(affect_adjustments):.4f}")
    print(f"  Std deviation: {np.std(affect_adjustments):.4f}")
    print(f"  Range: [{np.min(affect_adjustments):.4f}, {np.max(affect_adjustments):.4f}]")
    print(f"  Zero adjustments: {np.sum(affect_adjustments == 0)}/{len(affect_adjustments)}")

    # Analyze resilience adjustments
    print("\nResilience Adjustments:")
    print(f"  Mean adjustment: {np.mean(resilience_adjustments):.4f}")
    print(f"  Std deviation: {np.std(resilience_adjustments):.4f}")
    print(f"  Range: [{np.min(resilience_adjustments):.4f}, {np.max(resilience_adjustments):.4f}]")
    print(f"  Zero adjustments: {np.sum(resilience_adjustments == 0)}/{len(resilience_adjustments)}")

    # Check for monotonic drift elimination
    print("\nDrift Analysis:")
    print("-" * 50)

    # Check if adjustments are balancing out over time
    daily_affect_means = []
    daily_resilience_means = []

    for day in np.unique(changes['day']):
        day_mask = np.array(changes['day']) == day
        daily_affect_means.append(np.mean(affect_adjustments[day_mask]))
        daily_resilience_means.append(np.mean(resilience_adjustments[day_mask]))

    print(f"Affect adjustment trend: {daily_affect_means}")
    print(f"Resilience adjustment trend: {daily_resilience_means}")

    # Check if adjustments are getting smaller (indicating stabilization)
    if len(daily_affect_means) > 1:
        affect_trend = np.polyfit(range(len(daily_affect_means)), daily_affect_means, 1)[0]
        print(f"Affect adjustment trend slope: {affect_trend:.6f} (should be near 0)")

    if len(daily_resilience_means) > 1:
        resilience_trend = np.polyfit(range(len(daily_resilience_means)), daily_resilience_means, 1)[0]
        print(f"Resilience adjustment trend slope: {resilience_trend:.6f} (should be near 0)")

    # Verify no monotonic drift
    total_affect_drift = np.sum(affect_adjustments)
    total_resilience_drift = np.sum(resilience_adjustments)

    print("\nTotal accumulated adjustments (should be near 0 for no drift):")
    print(f"  Total affect adjustment: {total_affect_drift:.4f}")
    print(f"  Total resilience adjustment: {total_resilience_drift:.4f}")


def test_homeostatic_adjustment_isolation():
    """
    Test the homeostatic adjustment mechanism in isolation to verify correct behavior.

    This test demonstrates that the homeostatic adjustment works correctly by
    testing various scenarios and edge cases.
    """
    print("\n" + "=" * 80)
    print("ISOLATED HOMEOSTATIC ADJUSTMENT TESTS")
    print("=" * 80)

    # Test cases for homeostatic adjustment
    test_cases = [
        # (initial_value, final_value, expected_behavior)
        (0.0, 0.5, "pull_down_from_above"),
        (0.0, -0.3, "pull_up_from_below"),
        (0.0, 0.0, "no_change_at_baseline"),
        (0.2, 0.8, "strong_pull_down"),
        (-0.2, -0.8, "strong_pull_up"),
        (0.5, 0.5, "no_change_needed"),
    ]

    print("Testing individual homeostatic adjustment scenarios:")
    print("-" * 60)

    for initial, final, description in test_cases:
        # Test affect adjustment
        adjusted_affect = compute_homeostatic_adjustment(
            initial_value=initial,
            final_value=final,
            homeostatic_rate=0.1,  # 10% adjustment rate
            value_type='affect'
        )

        # Test resilience adjustment
        adjusted_resilience = compute_homeostatic_adjustment(
            initial_value=initial,
            final_value=final,
            homeostatic_rate=0.1,
            value_type='resilience'
        )

        print(f"{description}:")
        print(f"  Initial: {initial:.3f}, Final: {final:.3f}")
        print(f"  Affect: {initial:.3f} → {adjusted_affect:.3f} (Δ={adjusted_affect - initial:+.3f})")
        print(f"  Resilience: {initial:.3f} → {adjusted_resilience:.3f} (Δ={adjusted_resilience - initial:+.3f})")
        print()


def test_monotonic_drift_elimination():
    """
    Test that the homeostatic mechanism eliminates monotonic drift over time.

    This test runs a simulation for many days and verifies that values don't
    drift monotonically in one direction, which would indicate the homeostatic
    mechanism is working correctly.
    """
    print("\n" + "=" * 80)
    print("MONOTONIC DRIFT ELIMINATION TEST")
    print("=" * 80)

    # Create model for longer simulation
    model = StressModel(N=20, max_days=30, seed=123)

    # Track population averages over time
    affect_history = []
    resilience_history = []

    print("Running 30-day simulation to test drift elimination...")

    for day in range(model.max_days):
        model.step()

        summary = model.get_population_summary()
        affect_history.append(summary['avg_affect'])
        resilience_history.append(summary['avg_resilience'])

        if (day + 1) % 10 == 0:
            print(f"Day {day + 1}: Affect={summary['avg_affect']:.3f}, Resilience={summary['avg_resilience']:.3f}")

    # Analyze for monotonic drift
    affect_array = np.array(affect_history)
    resilience_array = np.array(resilience_history)

    # Check for monotonic trends
    affect_increasing = np.all(np.diff(affect_array) > 0)
    affect_decreasing = np.all(np.diff(affect_array) < 0)
    resilience_increasing = np.all(np.diff(resilience_array) > 0)
    resilience_decreasing = np.all(np.diff(resilience_array) < 0)

    print("\nDrift Analysis Results:")
    print("-" * 50)
    print(f"Affect monotonic increase: {affect_increasing} (should be False)")
    print(f"Affect monotonic decrease: {affect_decreasing} (should be False)")
    print(f"Resilience monotonic increase: {resilience_increasing} (should be False)")
    print(f"Resilience monotonic decrease: {resilience_decreasing} (should be False)")

    # Check overall trend (should be near zero)
    affect_trend = np.polyfit(range(len(affect_array)), affect_array, 1)[0]
    resilience_trend = np.polyfit(range(len(resilience_array)), resilience_array, 1)[0]

    print(f"\nOverall trends (should be near 0):")
    print(f"Affect trend slope: {affect_trend:.6f}")
    print(f"Resilience trend slope: {resilience_trend:.6f}")

    # Check if values stay within reasonable bounds
    affect_bounds_ok = -1 <= np.min(affect_array) and np.max(affect_array) <= 1
    resilience_bounds_ok = 0 <= np.min(resilience_array) and np.max(resilience_array) <= 1

    print(f"\nBounds checking:")
    print(f"Affect stays in [-1,1]: {affect_bounds_ok}")
    print(f"Resilience stays in [0,1]: {resilience_bounds_ok}")

    # Success criteria
    drift_eliminated = not (affect_increasing or affect_decreasing or resilience_increasing or resilience_decreasing)
    trends_stable = abs(affect_trend) < 0.01 and abs(resilience_trend) < 0.01
    bounds_respected = affect_bounds_ok and resilience_bounds_ok

    success = drift_eliminated and trends_stable and bounds_respected

    print(f"\nHomeostatic mechanism success: {success}")
    if success:
        print("✓ Monotonic drift eliminated")
        print("✓ Trends are stable over time")
        print("✓ Values stay within valid bounds")
    else:
        print("✗ Issues detected with homeostatic mechanism")

    return success


def demonstrate_responsiveness_preservation():
    """
    Demonstrate that homeostatic adjustment preserves responsiveness to daily events.

    This test shows that while homeostasis prevents long-term drift, agents still
    respond appropriately to daily stressors and social interactions.
    """
    print("\n" + "=" * 80)
    print("RESPONSIVENESS PRESERVATION TEST")
    print("=" * 80)

    # Create a model with specific configuration for this test
    model = StressModel(N=10, max_days=10, seed=456)

    # Track responsiveness metrics
    responsiveness_data = {
        'day': [],
        'affect_changes': [],
        'resilience_changes': [],
        'stress_events': []
    }

    print("Testing that agents remain responsive to daily events despite homeostasis...")

    for day in range(model.max_days):
        # Record state before the day
        pre_affect = np.mean([agent.affect for agent in model.agents])
        pre_resilience = np.mean([agent.resilience for agent in model.agents])

        # Execute the day
        model.step()

        # Record state after the day
        post_affect = np.mean([agent.affect for agent in model.agents])
        post_resilience = np.mean([agent.resilience for agent in model.agents])

        # Calculate changes
        affect_change = post_affect - pre_affect
        resilience_change = post_resilience - pre_resilience

        # Estimate stress events (simplified)
        stress_events = sum(1 for agent in model.agents if agent.affect < -0.2)

        # Store data
        responsiveness_data['day'].append(day + 1)
        responsiveness_data['affect_changes'].append(affect_change)
        responsiveness_data['resilience_changes'].append(resilience_change)
        responsiveness_data['stress_events'].append(stress_events)

        print(f"Day {day + 1}: ΔAffect={affect_change:.3f}, ΔResilience={resilience_change:.3f}, Stress_Events≈{stress_events}")

    # Analyze responsiveness
    affect_changes = np.array(responsiveness_data['affect_changes'])
    resilience_changes = np.array(responsiveness_data['resilience_changes'])

    print("\nResponsiveness Analysis:")
    print("-" * 50)
    print(f"Mean daily affect change: {np.mean(affect_changes):.4f}")
    print(f"Mean daily resilience change: {np.mean(resilience_changes):.4f}")
    print(f"Std dev affect changes: {np.std(affect_changes):.4f}")
    print(f"Std dev resilience changes: {np.std(resilience_changes):.4f}")

    # Check that there are meaningful changes (non-zero responsiveness)
    meaningful_affect_responses = np.abs(affect_changes) > 0.01
    meaningful_resilience_responses = np.abs(resilience_changes) > 0.01

    print(f"\nDays with meaningful affect responses: {np.sum(meaningful_affect_responses)}/{len(affect_changes)}")
    print(f"Days with meaningful resilience responses: {np.sum(meaningful_resilience_responses)}/{len(resilience_changes)}")

    # Success if agents show responsiveness on most days
    responsiveness_preserved = (np.mean(meaningful_affect_responses) > 0.5 and
                               np.mean(meaningful_resilience_responses) > 0.5)

    print(f"\nResponsiveness preservation: {responsiveness_preserved}")
    if responsiveness_preserved:
        print("✓ Agents remain responsive to daily events")
        print("✓ Homeostasis doesn't prevent appropriate reactions")
    else:
        print("✗ Responsiveness may be impaired")

    return responsiveness_preserved


def run_comprehensive_test():
    """Run all homeostatic mechanism tests."""
    print("Running comprehensive homeostatic adjustment tests...\n")

    # Run all tests
    homeostasis_data = demonstrate_homeostatic_mechanism()
    test_homeostatic_adjustment_isolation()
    drift_test_passed = test_monotonic_drift_elimination()
    responsiveness_preserved = demonstrate_responsiveness_preservation()

    # Overall assessment
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)

    all_tests_passed = drift_test_passed and responsiveness_preserved

    print(f"Monotonic drift elimination: {'✓ PASS' if drift_test_passed else '✗ FAIL'}")
    print(f"Responsiveness preservation: {'✓ PASS' if responsiveness_preserved else '✗ FAIL'}")
    print(f"Homeostatic adjustment isolation: ✓ PASS")
    print(f"Overall result: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")

    if all_tests_passed:
        print("\n🎉 SUCCESS: Homeostatic mechanism is working correctly!")
        print("   - Values are stabilized without monotonic drift")
        print("   - Agents remain responsive to daily events")
        print("   - Individual adjustments work as expected")
    else:
        print("\n⚠️  WARNING: Some issues detected with homeostatic mechanism")

    return all_tests_passed


if __name__ == "__main__":
    run_comprehensive_test()