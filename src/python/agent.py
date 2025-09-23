# Import modules

import random
import numpy as np
import pandas as pd
import mesa

# Function

def influence(affect, diff = 0.05):
    changes = float(diff * np.sign(affect))
    return changes

def clamp(value, minval=0, maxval=1):
    clean = min(max(value, minval), maxval)
    return clean

# Initialize an agent

class Person(mesa.Agent):
    """
    A person who experiences social interactions and stressful events.
    Positive/negative affect is tracked in self.affect.
    """

    def __init__(self, model):
        # Pas the parameters to the parent class
        super().__init__(model)

        # Set an agent variable
        self.resilience = 0.5
        self.affect = 0

    def step(self):
        """
        One day of simulation epoch. Perform a random sequence of interactions
        and stressful events in random order and random count.
        """
        # Decide the number of subevents
        n_subevents = np.random.poisson(lam=3) # 3 subevents per day
        if n_subevents == 0:
            n_subevents = 1

        # Randomly choose the event type for each subevent
        actions = [
            random.choice(["interact", "stress"]) for _ in range(n_subevents)
        ]

        # Shuffle to create a random order
        random.shuffle(actions)

        # Execute actions
        for act in actions:
            if act == "interact":
                self.interact()
            else:
                self.stressful_event()

        # Clamp values to prevent moving beyond the reasonable bound
        self.resilience = clamp(self.resilience, minval=0, maxval=1)
        self.affect = clamp(self.affect, minval=-1, maxval=1)

    def interact(self):
        """
        Interact with a random neighbor. The agent's affect shifts slightly
        toward the neighbor's affect.
        """
        neighbors = list(
            self.model.grid.get_neighbors(
                self.pos, include_center=False
            )
        )
        if not neighbors:
            return

        partner = random.choice(neighbors)

        # Positive neighbor pulls affect upward, negative pulls downward
        self.affect += influence(partner.affect, diff = 0.05)
        partner.affect += influence(self.affect, diff = 0.05)
        self.resilience += influence(partner.affect, diff = 0.05)
        partner.resilience += influence(self.affect, diff = 0.05)

        # Clamp the values to prevent overflow
        self.affect = clamp(self.affect, minval=-1, maxval=1)
        partner.affect = clamp(partner.affect, minval=-1, maxval=1)
        self.resilience = clamp(self.resilience, minval=-1, maxval=1)
        partner.resilience = clamp(partner.resilience, minval=-1, maxval=1)

    def stressful_event(self):
        """
        Chance of stress. If stressed, coping depends on resilience. Coping
        improves resilience and affect, failure reduces both.
        """
        stressed = random.random() < 0.5
        if not stressed:
            return

        coped = random.random() < self.resilience
        if coped:
            self.resilience += 0.05
            self.affect += 0.05
        else:
            self.resilience -= 0.05
            self.affect -= 0.05
