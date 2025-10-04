import sys
import numpy as np
import pandas as pd

sys.path.append('.')

from src.python.model import StressModel

# Create a test with more detailed debugging
model = StressModel(N=5, max_days=5, seed=42)

print("Starting simulation...")

for step in range(5):
    print(f"\n=== DAY {step} ===")
    
    # Check RNG state and agent configuration before step
    for i, agent in enumerate(model.agents):
        print(f"Agent {i}:")
        print(f"  - RNG type: {type(agent._rng)}")
        print(f"  - Last reset day: {agent.last_reset_day}")
        print(f"  - Current stress: {agent.current_stress}")
        print(f"  - Events before: {len(agent.daily_stress_events)}")
        
        # Test RNG directly
        test_actions = [agent._rng.choice(["interact", "stress"]) for _ in range(5)]
        stress_count = sum(1 for action in test_actions if action == "stress")
        print(f"  - Test RNG (5 draws): {stress_count} stress actions")
    
    # Execute step
    model.step()
    
    # Check results
    for i, agent in enumerate(model.agents):
        events_after = len(agent.daily_stress_events)
        print(f"Agent {i} - Events after: {events_after}")
        
        if events_after > 0:
            for j, event in enumerate(agent.daily_stress_events):
                print(f"  Event {j}: challenge={event.get('challenge', 0):.3f}, hindrance={event.get('hindrance', 0):.3f}")

# Check final DataCollector data
print("\n" + "="*50)
print("FINAL DATACOLLECTOR DATA:")
tbl = model.get_time_series_data()
print(tbl[['total_stress_events', 'stress_events']])
