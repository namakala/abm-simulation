# Import modules

import networkx as nx
import numpy as np
import pandas as pd
import mesa

from mesa.space import NetworkGrid
from src.python.agent import Person
from src.python.config import get_config

# Load configuration
config = get_config()

# Initialize the model

class StressModel(mesa.Model):
    """
    Model description
    """

    def __init__(self, N=None, max_days=None, seed=None):
        super().__init__(seed=seed)
        self.day = 0

        # Use config values if parameters not provided
        if N is None:
            N = config.get('simulation', 'num_agents')
        if max_days is None:
            max_days = config.get('simulation', 'max_days')
        if seed is None:
            seed = config.get('simulation', 'seed')

        self.max_days = max_days

        # Build social network
        G = nx.watts_strogatz_graph(
            n=N,
            k=config.get('network', 'watts_k'),
            p=config.get('network', 'watts_p')
        )
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

