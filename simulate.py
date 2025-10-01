# Import modules

from src.python.agent import Person
from src.python.model import StressModel
from src.python.config import get_config

# Load configuration
config = get_config()

# Simulate
model = StressModel(
    N=config.get('simulation', 'num_agents'),
    max_days=config.get('simulation', 'max_days'),
    seed=config.get('simulation', 'seed')
)

while model.running:
    model.step()

resilience = [a.resilience for a in model.agents]
print(resilience)

affect = [a.affect for a in model.agents]
print(affect)

