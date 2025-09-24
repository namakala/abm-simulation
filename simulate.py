# Import modules

from src.python.agent import Person
from src.python.model import StressModel

# Simulate
model = StressModel(N=10, max_days=1000, seed=42)
while model.running:
    model.step()

resilience = [a.resilience for a in model.agents]
print(resilience)

affect = [a.affect for a in model.agents]
print(affect)

