"""Test script to verify simulation runs correctly with realistic behavior."""

from src.python.model import StressModel
from src.python.config import Config

def test_simulation():
    """Test that simulation runs correctly over multiple days."""
    print("Testing simulation behavior over 10 days...")

    config = Config()
    model = StressModel(max_days=10)

    print("Running 10-day simulation...")
    for i in range(10):
        model.step()
        summary = model.get_population_summary()
        print(f"Day {model.day}: Affect={summary['avg_affect']:.3f}, Resilience={summary['avg_resilience']:.3f}, Stress prevalence={summary['stress_prevalence']:.3f}")

    print("Simulation completed successfully!")

    # Verify realistic behavior
    final_summary = model.get_population_summary()

    # Check that values are in reasonable ranges
    assert -1 <= final_summary['avg_affect'] <= 1, f"Affect out of range: {final_summary['avg_affect']}"
    assert 0 <= final_summary['avg_resilience'] <= 1, f"Resilience out of range: {final_summary['avg_resilience']}"
    assert 0 <= final_summary['stress_prevalence'] <= 1, f"Stress prevalence out of range: {final_summary['stress_prevalence']}"

    print("✓ All values are in realistic ranges")
    print("✓ Simulation behavior is realistic")

if __name__ == "__main__":
    test_simulation()