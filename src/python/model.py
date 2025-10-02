# Import modules

import networkx as nx
import numpy as np
import pandas as pd
import mesa
from typing import Dict, List, Any
from collections import defaultdict

from mesa.space import NetworkGrid
from src.python.agent import Person
from src.python.config import get_config

# Load configuration
config = get_config()

# Initialize the model

class StressModel(mesa.Model):
    """
    Enhanced agent-based model for mental health simulation with social interactions,
    affect dynamics, and resilience tracking.

    Supports:
    - Social interactions between neighboring agents
    - Affect and resilience dynamics at both individual and population levels
    - Social support exchange tracking
    - Population-level metrics collection
    - Enhanced agent capabilities with protective factors and network adaptation
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
        self.num_agents = N

        # Initialize data collection for population metrics
        self._initialize_data_collection()

        # Build social network
        G = nx.watts_strogatz_graph(
            n=N,
            k=config.get('network', 'watts_k'),
            p=config.get('network', 'watts_p')
        )
        self.grid = NetworkGrid(G)

        # Create and register agents with enhanced capabilities
        for node in G.nodes():
            agent = Person(self)
            self.agents.add(agent)
            self.grid.place_agent(agent, node)

        # Initialize social support tracking
        self.social_support_exchanges = 0
        self.total_interactions = 0

        # Initialize population statistics tracking with enhanced metrics
        self.population_metrics = {
            'day': [],
            'avg_affect': [],
            'avg_resilience': [],
            'avg_resources': [],
            'social_support_rate': [],
            'stress_events': [],
            'network_density': [],
            'stress_prevalence': [],
            'low_resilience': [],
            'high_resilience': []
        }

        self.running = True

    def _initialize_data_collection(self):
        """Initialize data collection structures for population metrics."""
        # Track daily statistics
        self.daily_stats = {
            'total_stress_events': 0,
            'successful_coping': 0,
            'social_interactions': 0,
            'support_exchanges': 0
        }

        # Track agent-level time series if needed
        self.agent_time_series = defaultdict(lambda: {
            'affect': [], 'resilience': [], 'resources': []
        })

    def step(self):
        """
        Execute one day of simulation with enhanced social interactions and dynamics.

        Process order:
        1. Reset daily statistics
        2. Execute agent steps (interactions and stress events)
        3. Collect population-level metrics
        4. Update social support tracking
        5. Record daily statistics
        """
        # Reset daily statistics
        self._reset_daily_stats()

        # Execute agent steps with enhanced social interactions
        self.agents.shuffle_do("step")

        # Collect and record population metrics
        self._collect_population_metrics()

        # Update social support tracking
        self._update_social_support_tracking()

        # Record daily statistics
        self._record_daily_stats()

        # Increment day counter
        self.day += 1

        # Check termination condition
        if self.day >= self.max_days:
            self.running = False

    def _reset_daily_stats(self):
        """Reset daily statistics tracking."""
        self.daily_stats = {
            'total_stress_events': 0,
            'successful_coping': 0,
            'social_interactions': 0,
            'support_exchanges': 0
        }

    def _collect_population_metrics(self):
        """Collect population-level metrics for affect, resilience, and resources."""
        if not self.agents:
            return

        # Calculate population averages with enhanced metrics
        total_affect = sum(agent.affect for agent in self.agents)
        total_resilience = sum(agent.resilience for agent in self.agents)
        total_resources = sum(agent.resources for agent in self.agents)

        avg_affect = total_affect / len(self.agents)
        avg_resilience = total_resilience / len(self.agents)
        avg_resources = total_resources / len(self.agents)

        # Enhanced social support rate calculation
        social_support_rate = self._calculate_social_support_rate()

        # Enhanced stress events counting with challenge/hindrance awareness
        stress_events = self._count_stress_events()

        # Calculate network density
        network_density = self._calculate_network_density()

        # Calculate additional integrated metrics
        # Stress prevalence based on affect threshold
        stressed_agents   = sum(1 for agent in self.agents if agent.affect < -0.3)
        stress_prevalence = stressed_agents / len(self.agents)

        # Resilience distribution for population health assessment
        low_resilience  = sum(1 for agent in self.agents if agent.resilience < 0.3)
        high_resilience = sum(1 for agent in self.agents if agent.resilience > 0.7)

        # Store enhanced metrics
        self.population_metrics['day'].append(self.day)
        self.population_metrics['avg_affect'].append(avg_affect)
        self.population_metrics['avg_resilience'].append(avg_resilience)
        self.population_metrics['avg_resources'].append(avg_resources)
        self.population_metrics['social_support_rate'].append(social_support_rate)
        self.population_metrics['stress_events'].append(stress_events)
        self.population_metrics['network_density'].append(network_density)
        self.population_metrics['stress_prevalence'].append(stress_prevalence)
        self.population_metrics['low_resilience'].append(low_resilience)
        self.population_metrics['high_resilience'].append(high_resilience)

    def _calculate_social_support_rate(self) -> float:
        """Calculate rate of social support exchanges in the population."""
        if self.total_interactions == 0:
            return 0.0

        return self.social_support_exchanges / self.total_interactions

    def _count_stress_events(self) -> int:
        """Count approximate number of stress events from agent states."""
        # This is an approximation - in a full implementation, agents would report events
        stressed_agents = sum(1 for agent in self.agents if agent.affect < -0.2)
        return int(stressed_agents * 0.3)  # Assume 30% of stressed agents had events

    def _calculate_network_density(self) -> float:
        """Calculate approximate network density based on agent connections."""
        if not self.agents:
            return 0.0

        # Count actual connections in the NetworkX graph
        total_possible_connections = self.num_agents * (self.num_agents - 1)
        if total_possible_connections == 0:
            return 0.0

        actual_connections = sum(1 for _ in self.grid.G.edges())
        return actual_connections / total_possible_connections

    def _update_social_support_tracking(self):
        """Update tracking of social support exchanges between agents."""
        # This would be enhanced to track actual support exchanges
        # For now, we estimate based on interactions
        pass

    def _record_daily_stats(self):
        """Record daily statistics for analysis."""
        # Update cumulative counters
        self.total_interactions += self.daily_stats['social_interactions']
        self.social_support_exchanges += self.daily_stats['support_exchanges']

    def get_population_summary(self) -> Dict[str, Any]:
        """
        Get summary of current population state.

        Returns:
            Dictionary containing population metrics and statistics
        """
        if not self.agents:
            return {}

        # Current averages
        current_affect = np.mean([agent.affect for agent in self.agents])
        current_resilience = np.mean([agent.resilience for agent in self.agents])
        current_resources = np.mean([agent.resources for agent in self.agents])

        # Distribution statistics
        affect_std = np.std([agent.affect for agent in self.agents])
        resilience_std = np.std([agent.resilience for agent in self.agents])

        # Stress prevalence
        stressed_agents = sum(1 for agent in self.agents if agent.affect < -0.3)
        stress_prevalence = stressed_agents / len(self.agents)

        # Resilience categories
        low_resilience = sum(1 for agent in self.agents if agent.resilience < 0.3)
        medium_resilience = sum(1 for agent in self.agents
                              if 0.3 <= agent.resilience <= 0.7)
        high_resilience = sum(1 for agent in self.agents if agent.resilience > 0.7)

        return {
            'day': self.day,
            'num_agents': len(self.agents),
            'avg_affect': current_affect,
            'affect_std': affect_std,
            'avg_resilience': current_resilience,
            'resilience_std': resilience_std,
            'avg_resources': current_resources,
            'stress_prevalence': stress_prevalence,
            'resilience_distribution': {
                'low': low_resilience,
                'medium': medium_resilience,
                'high': high_resilience
            },
            'social_support_rate': self._calculate_social_support_rate(),
            'total_interactions': self.total_interactions,
            'network_density': self._calculate_network_density(),
            # Enhanced integrated metrics
            'mental_health_index': (current_affect + current_resilience) / 2,  # Combined mental health score
            'recovery_potential': high_resilience / max(1, len(self.agents)),  # Proportion with high resilience
            'vulnerability_index': low_resilience / max(1, len(self.agents))   # Proportion with low resilience
        }

    def get_time_series_data(self) -> pd.DataFrame:
        """
        Get time series data for population metrics.

        Returns:
            DataFrame with daily population metrics
        """
        if not self.population_metrics['day']:
            return pd.DataFrame()

        return pd.DataFrame(self.population_metrics)

    def export_results(self, filename: str = None) -> str:
        """
        Export simulation results to CSV file.

        Args:
            filename: Optional filename for export (default: auto-generated)

        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"simulation_results_day_{self.day}.csv"

        df = self.get_time_series_data()

        if not df.empty:
            df.to_csv(filename, index=False)

        return filename

