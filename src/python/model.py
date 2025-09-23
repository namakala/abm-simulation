# Import modules

import networkx as nx
import numpy as np
import pandas as pd
import mesa

from mesa.space import NetworkGrid
from src.python.agent import Person

# Initialize the model

class StressModel(mesa.Model):
    """
    Model description
    """

    def __init__(self, N=20, max_days=100, seed=None):
        super().__init__(seed=seed)
        self.day = 0
        self.max_days = max_days

        # Build social network
        G = nx.watts_strogatz_graph(n=N, k=4, p=0.1)
        self.grid = NetworkGrid(G)

        # Create and register agents
        for node in G.nodes():
            agent = Person(self)
            self.agents.add(agent)
            self.grid.place_agent(agent, node)

        self.running = True

    def step(self):
        self.agents.shuffle_do("step")
        self.day += 1
        if self.day >= self.max_days:
            self.running = False

# Simulate
model = StressModel(N=10, max_days=1000, seed=42)
while model.running:
    model.step()

resilience = [a.resilience for a in model.agents]
print(resilience)

affect = [a.affect for a in model.agents]
print(affect)
