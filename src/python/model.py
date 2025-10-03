# Import modules

import networkx as nx
import numpy as np
import pandas as pd
import mesa
from typing import Dict, List, Any
from collections import defaultdict

from mesa.space import NetworkGrid
from mesa import DataCollector
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

        # Initialize DataCollector for population and agent metrics
        self._initialize_datacollector()

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
            'avg_stress': [],
            'social_support_rate': [],
            'stress_events': [],
            'network_density': [],
            'stress_prevalence': [],
            'low_resilience': [],
            'high_resilience': [],
            # New stress processing metrics
            'avg_challenge': [],
            'avg_hindrance': [],
            'coping_success_rate': [],
            'avg_consecutive_hindrances': [],
            'challenge_hindrance_ratio': []
        }

        self.running = True

    def _initialize_datacollector(self):
        """Initialize Mesa DataCollector for population and agent metrics."""
        # Define model reporters (population-level metrics)
        model_reporters = {
            'avg_pss10': lambda m: m.get_avg_pss10(),
            'avg_resilience': lambda m: m.get_avg_resilience(),
            'avg_affect': lambda m: m.get_avg_affect(),
            'coping_success_rate': lambda m: m.get_success_rate(),
            'avg_resources': lambda m: np.mean([agent.resources for agent in m.agents]) if m.agents else 0.0,
            'avg_stress': lambda m: np.mean([getattr(agent, 'current_stress', 0.0) for agent in m.agents]) if m.agents else 0.0,
            'social_support_rate': lambda m: m._calculate_social_support_rate(),
            'stress_events': lambda m: sum(len(getattr(agent, 'daily_stress_events', [])) for agent in m.agents),
            'network_density': lambda m: m._calculate_network_density(),
            'stress_prevalence': lambda m: sum(1 for agent in m.agents if agent.affect < -0.3) / len(m.agents) if m.agents else 0.0,
            'low_resilience': lambda m: sum(1 for agent in m.agents if agent.resilience < 0.3) if m.agents else 0,
            'high_resilience': lambda m: sum(1 for agent in m.agents if agent.resilience > 0.7) if m.agents else 0,
            'avg_challenge': lambda m: m._get_avg_challenge(),
            'avg_hindrance': lambda m: m._get_avg_hindrance(),
            'challenge_hindrance_ratio': lambda m: m._get_challenge_hindrance_ratio(),
            'avg_consecutive_hindrances': lambda m: m._get_avg_consecutive_hindrances()
        }

        # Define agent reporters (agent-level metrics)
        agent_reporters = {
            'pss10': lambda a: a.pss10,
            'resilience': lambda a: a.resilience,
            'affect': lambda a: a.affect,
            'resources': lambda a: a.resources,
            'current_stress': lambda a: getattr(a, 'current_stress', 0.0),
            'stress_controllability': lambda a: getattr(a, 'stress_controllability', 0.5),
            'stress_overload': lambda a: getattr(a, 'stress_overload', 0.5),
            'consecutive_hindrances': lambda a: getattr(a, 'consecutive_hindrances', 0)
        }

        # Initialize DataCollector
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters
        )

        # Keep daily stats for backward compatibility
        self.daily_stats = {
            'total_stress_events': 0,
            'successful_coping': 0,
            'social_interactions': 0,
            'support_exchanges': 0
        }

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

        # Collect data using DataCollector
        self.datacollector.collect(self)

        # Apply network adaptation mechanisms
        self._apply_network_adaptation()

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
        """Collect population-level metrics for affect, resilience, resources, and new stress processing metrics."""
        if not self.agents:
            return

        # Calculate population averages with enhanced metrics
        total_affect = sum(agent.affect for agent in self.agents)
        total_resilience = sum(agent.resilience for agent in self.agents)
        total_resources = sum(agent.resources for agent in self.agents)
        total_stress = sum(getattr(agent, 'current_stress', 0.0) for agent in self.agents)

        avg_affect = total_affect / len(self.agents)
        avg_resilience = total_resilience / len(self.agents)
        avg_resources = total_resources / len(self.agents)
        avg_stress = total_stress / len(self.agents)

        # Enhanced social support rate calculation
        social_support_rate = self._calculate_social_support_rate()

        # Enhanced stress events counting with actual data from agents
        stress_events_data = self._collect_stress_events_data()

        # Calculate network density
        network_density = self._calculate_network_density()

        # Calculate additional integrated metrics
        # Stress prevalence based on affect threshold
        stressed_agents   = sum(1 for agent in self.agents if agent.affect < -0.3)
        stress_prevalence = stressed_agents / len(self.agents)

        # Resilience distribution for population health assessment
        low_resilience  = sum(1 for agent in self.agents if agent.resilience < 0.3)
        high_resilience = sum(1 for agent in self.agents if agent.resilience > 0.7)

        # New stress processing metrics
        challenge_hindrance_data = self._collect_challenge_hindrance_data()
        coping_data = self._collect_coping_data()
        consecutive_hindrances_data = self._collect_consecutive_hindrances_data()

        # Store enhanced metrics
        self.population_metrics['day'].append(self.day)
        self.population_metrics['avg_affect'].append(avg_affect)
        self.population_metrics['avg_resilience'].append(avg_resilience)
        self.population_metrics['avg_resources'].append(avg_resources)
        self.population_metrics['avg_stress'].append(avg_stress)
        self.population_metrics['social_support_rate'].append(social_support_rate)
        self.population_metrics['stress_events'].append(stress_events_data['count'])
        self.population_metrics['network_density'].append(network_density)
        self.population_metrics['stress_prevalence'].append(stress_prevalence)
        self.population_metrics['low_resilience'].append(low_resilience)
        self.population_metrics['high_resilience'].append(high_resilience)

        # Store new stress processing metrics
        self.population_metrics['avg_challenge'].append(challenge_hindrance_data['avg_challenge'])
        self.population_metrics['avg_hindrance'].append(challenge_hindrance_data['avg_hindrance'])
        self.population_metrics['coping_success_rate'].append(coping_data['success_rate'])
        self.population_metrics['avg_consecutive_hindrances'].append(consecutive_hindrances_data['avg'])
        self.population_metrics['challenge_hindrance_ratio'].append(challenge_hindrance_data['ratio'])

    def _calculate_social_support_rate(self) -> float:
        """Calculate rate of social support exchanges in the population."""
        if self.total_interactions == 0:
            return 0.0

        return self.social_support_exchanges / self.total_interactions

    def _collect_stress_events_data(self) -> Dict[str, Any]:
        """Collect actual stress events data from all agents."""
        total_events = 0
        total_challenge = 0.0
        total_hindrance = 0.0

        for agent in self.agents:
            # Count events from agent's daily stress events
            daily_events = getattr(agent, 'daily_stress_events', [])
            agent_events = len(daily_events)
            total_events += agent_events

            # Sum challenge and hindrance from events
            for event in daily_events:
                total_challenge += event.get('challenge', 0.0)
                total_hindrance += event.get('hindrance', 0.0)

        # Calculate averages
        if total_events > 0:
            avg_challenge = total_challenge / total_events
            avg_hindrance = total_hindrance / total_events
        else:
            avg_challenge = 0.0
            avg_hindrance = 0.0

        return {
            'count': total_events,
            'avg_challenge': avg_challenge,
            'avg_hindrance': avg_hindrance
        }

    def _collect_challenge_hindrance_data(self) -> Dict[str, float]:
        """Collect population-level challenge and hindrance statistics."""
        total_challenge = 0.0
        total_hindrance = 0.0
        total_events = 0

        for agent in self.agents:
            daily_events = getattr(agent, 'daily_stress_events', [])
            for event in daily_events:
                total_challenge += event.get('challenge', 0.0)
                total_hindrance += event.get('hindrance', 0.0)
                total_events += 1

        if total_events > 0:
            avg_challenge = total_challenge / total_events
            avg_hindrance = total_hindrance / total_events
            # Calculate ratio (challenge dominance vs hindrance dominance)
            ratio = (avg_challenge - avg_hindrance) if (avg_challenge + avg_hindrance) > 0 else 0.0
        else:
            avg_challenge = 0.0
            avg_hindrance = 0.0
            ratio = 0.0

        return {
            'avg_challenge': avg_challenge,
            'avg_hindrance': avg_hindrance,
            'ratio': ratio
        }

    def _collect_coping_data(self) -> Dict[str, float]:
        """Collect population-level coping success statistics."""
        total_attempts = 0
        total_successes = 0

        for agent in self.agents:
            daily_events = getattr(agent, 'daily_stress_events', [])
            for event in daily_events:
                if 'coped_successfully' in event:
                    total_attempts += 1
                    if event['coped_successfully']:
                        total_successes += 1

        if total_attempts > 0:
            success_rate = total_successes / total_attempts
        else:
            success_rate = 0.0

        return {
            'success_rate': success_rate,
            'total_attempts': total_attempts,
            'total_successes': total_successes
        }

    def _collect_consecutive_hindrances_data(self) -> Dict[str, float]:
        """Collect population-level consecutive hindrances statistics."""
        total_consecutive = 0.0
        valid_agents = 0

        for agent in self.agents:
            consecutive = getattr(agent, 'consecutive_hindrances', 0)
            if consecutive > 0:  # Only count agents with hindrance tracking
                total_consecutive += consecutive
                valid_agents += 1

        if valid_agents > 0:
            avg_consecutive = total_consecutive / valid_agents
        else:
            avg_consecutive = 0.0

        return {
            'avg': avg_consecutive,
            'max': max((getattr(agent, 'consecutive_hindrances', 0) for agent in self.agents), default=0.0),
            'agents_affected': valid_agents
        }

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

    def _apply_network_adaptation(self):
        """Apply network adaptation mechanisms across all agents."""
        # This method coordinates network adaptation at the model level
        # Individual agents handle their own adaptation in their step() method
        # but the model can track population-level adaptation metrics

        adaptation_count = 0
        for agent in self.agents:
            # Check if agent performed network adaptation (tracked via attribute)
            if hasattr(agent, '_adapted_network') and agent._adapted_network:
                adaptation_count += 1
                agent._adapted_network = False  # Reset for next day

        return adaptation_count

    def get_network_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of network adaptation across the population."""
        total_adaptations = 0
        agents_adapting = 0

        for agent in self.agents:
            # Count agents who have adapted their network recently
            stress_breach_count = getattr(agent, 'stress_breach_count', 0)
            if stress_breach_count >= 3:  # Adaptation threshold
                agents_adapting += 1

        return {
            'agents_considering_adaptation': agents_adapting,
            'adaptation_rate': agents_adapting / max(1, len(self.agents))
        }

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

        # Use DataCollector if available, otherwise calculate manually
        if hasattr(self, 'datacollector') and self.datacollector is not None:
            try:
                # Get latest data from DataCollector
                model_data = self.datacollector.get_model_vars_dataframe()
                if not model_data.empty:
                    latest_data = model_data.iloc[-1] if len(model_data) > 0 else {}

                    # Get agent data for additional statistics
                    agent_data = self.datacollector.get_agent_vars_dataframe()

                    # Current averages from DataCollector
                    current_affect = latest_data.get('avg_affect', np.mean([agent.affect for agent in self.agents]))
                    current_resilience = latest_data.get('avg_resilience', np.mean([agent.resilience for agent in self.agents]))
                    current_resources = latest_data.get('avg_resources', np.mean([agent.resources for agent in self.agents]))
                    current_stress = latest_data.get('avg_stress', np.mean([getattr(agent, 'current_stress', 0.0) for agent in self.agents]))

                    # Distribution statistics
                    if not agent_data.empty:
                        affect_std = agent_data['affect'].std() if 'affect' in agent_data.columns else np.std([agent.affect for agent in self.agents])
                        resilience_std = agent_data['resilience'].std() if 'resilience' in agent_data.columns else np.std([agent.resilience for agent in self.agents])
                        stress_std = agent_data['current_stress'].std() if 'current_stress' in agent_data.columns else np.std([getattr(agent, 'current_stress', 0.0) for agent in self.agents])
                    else:
                        affect_std = np.std([agent.affect for agent in self.agents])
                        resilience_std = np.std([agent.resilience for agent in self.agents])
                        stress_std = np.std([getattr(agent, 'current_stress', 0.0) for agent in self.agents])

                    # Stress prevalence
                    stressed_agents = latest_data.get('stress_prevalence', sum(1 for agent in self.agents if agent.affect < -0.3))
                    stress_prevalence = stressed_agents * len(self.agents) if isinstance(stressed_agents, float) else stressed_agents / len(self.agents)

                    # Resilience categories
                    low_resilience = latest_data.get('low_resilience', sum(1 for agent in self.agents if agent.resilience < 0.3))
                    high_resilience = latest_data.get('high_resilience', sum(1 for agent in self.agents if agent.resilience > 0.7))
                    medium_resilience = len(self.agents) - low_resilience - high_resilience

                    return {
                        'day': self.day,
                        'num_agents': len(self.agents),
                        'avg_affect': current_affect,
                        'affect_std': affect_std,
                        'avg_resilience': current_resilience,
                        'resilience_std': resilience_std,
                        'avg_resources': current_resources,
                        'avg_stress': current_stress,
                        'stress_std': stress_std,
                        'stress_prevalence': stress_prevalence,
                        'resilience_distribution': {
                            'low': low_resilience,
                            'medium': medium_resilience,
                            'high': high_resilience
                        },
                        'social_support_rate': latest_data.get('social_support_rate', self._calculate_social_support_rate()),
                        'total_interactions': self.total_interactions,
                        'network_density': latest_data.get('network_density', self._calculate_network_density()),
                        # Enhanced integrated metrics
                        'mental_health_index': (current_affect + current_resilience) / 2,  # Combined mental health score
                        'recovery_potential': high_resilience / max(1, len(self.agents)),  # Proportion with high resilience
                        'vulnerability_index': low_resilience / max(1, len(self.agents)),   # Proportion with low resilience
                        # New stress processing metrics
                        'avg_challenge': latest_data.get('avg_challenge', self._get_avg_challenge()),
                        'avg_hindrance': latest_data.get('avg_hindrance', self._get_avg_hindrance()),
                        'challenge_hindrance_ratio': latest_data.get('challenge_hindrance_ratio', self._get_challenge_hindrance_ratio()),
                        'coping_success_rate': latest_data.get('success_rate', self.get_success_rate()),
                        'avg_consecutive_hindrances': latest_data.get('avg_consecutive_hindrances', self._get_avg_consecutive_hindrances()),
                        'max_consecutive_hindrances': 0,  # Would need additional calculation
                        'agents_with_hindrances': 0      # Would need additional calculation
                    }
            except Exception:
                # Fallback to manual calculation if DataCollector fails
                pass

        # Manual calculation (fallback)
        # Current averages
        current_affect = np.mean([agent.affect for agent in self.agents])
        current_resilience = np.mean([agent.resilience for agent in self.agents])
        current_resources = np.mean([agent.resources for agent in self.agents])
        current_stress = np.mean([getattr(agent, 'current_stress', 0.0) for agent in self.agents])

        # Distribution statistics
        affect_std = np.std([agent.affect for agent in self.agents])
        resilience_std = np.std([agent.resilience for agent in self.agents])
        stress_std = np.std([getattr(agent, 'current_stress', 0.0) for agent in self.agents])

        # Stress prevalence
        stressed_agents = sum(1 for agent in self.agents if agent.affect < -0.3)
        stress_prevalence = stressed_agents / len(self.agents)

        # Resilience categories
        low_resilience = sum(1 for agent in self.agents if agent.resilience < 0.3)
        medium_resilience = sum(1 for agent in self.agents
                              if 0.3 <= agent.resilience <= 0.7)
        high_resilience = sum(1 for agent in self.agents if agent.resilience > 0.7)

        # New stress processing metrics
        challenge_hindrance_data = self._collect_challenge_hindrance_data()
        coping_data = self._collect_coping_data()
        consecutive_hindrances_data = self._collect_consecutive_hindrances_data()

        return {
            'day': self.day,
            'num_agents': len(self.agents),
            'avg_affect': current_affect,
            'affect_std': affect_std,
            'avg_resilience': current_resilience,
            'resilience_std': resilience_std,
            'avg_resources': current_resources,
            'avg_stress': current_stress,
            'stress_std': stress_std,
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
            'vulnerability_index': low_resilience / max(1, len(self.agents)),   # Proportion with low resilience
            # New stress processing metrics
            'avg_challenge': challenge_hindrance_data['avg_challenge'],
            'avg_hindrance': challenge_hindrance_data['avg_hindrance'],
            'challenge_hindrance_ratio': challenge_hindrance_data['ratio'],
            'coping_success_rate': coping_data['success_rate'],
            'avg_consecutive_hindrances': consecutive_hindrances_data['avg'],
            'max_consecutive_hindrances': consecutive_hindrances_data['max'],
            'agents_with_hindrances': consecutive_hindrances_data['agents_affected']
        }

    def get_time_series_data(self) -> pd.DataFrame:
        """
        Get time series data for population metrics.

        Returns:
            DataFrame with daily population metrics
        """
        # Use DataCollector to get model data
        if not hasattr(self, 'datacollector') or self.datacollector is None:
            # Fallback to old method if DataCollector not initialized
            if not self.population_metrics['day']:
                return pd.DataFrame()
            return pd.DataFrame(self.population_metrics)

        # Get data from DataCollector
        model_data = self.datacollector.get_model_vars_dataframe()
        if model_data.empty:
            return pd.DataFrame()

        return model_data
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

        # Get data from DataCollector
        df = self.get_time_series_data()

        if not df.empty:
            df.to_csv(filename, index=False)

        return filename

    def export_agent_data(self, filename: str = None) -> str:
        """
        Export agent-level time series data to CSV file.

        Args:
            filename: Optional filename for export (default: auto-generated)

        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"agent_data_day_{self.day}.csv"

        # Get agent data from DataCollector
        df = self.get_agent_time_series_data()

        if not df.empty:
            df.to_csv(filename, index=False)

        return filename

    def get_avg_pss10(self) -> float:
        """Calculate population average PSS-10 score."""
        if not self.agents:
            return 0.0

        try:
            # Handle missing PSS-10 data gracefully
            pss10_values = []
            for agent in self.agents:
                if hasattr(agent, 'pss10') and agent.pss10 is not None:
                    pss10_values.append(float(agent.pss10))
                else:
                    # Use default PSS-10 score if missing
                    pss10_values.append(10.0)

            if not pss10_values:
                return 0.0

            return sum(pss10_values) / len(pss10_values)
        except (AttributeError, TypeError, ValueError):
            # Fallback to 0.0 if there are any errors
            return 0.0

    def get_avg_resilience(self) -> float:
        """Calculate population average resilience."""
        if not self.agents:
            return 0.0

        try:
            resilience_values = [float(agent.resilience) for agent in self.agents if hasattr(agent, 'resilience')]
            if not resilience_values:
                return 0.0
            return sum(resilience_values) / len(resilience_values)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def get_avg_affect(self) -> float:
        """Calculate population average affect."""
        if not self.agents:
            return 0.0

        try:
            affect_values = [float(agent.affect) for agent in self.agents if hasattr(agent, 'affect')]
            if not affect_values:
                return 0.0
            return sum(affect_values) / len(affect_values)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def get_success_rate(self) -> float:
        """Calculate population coping success rate."""
        if not self.agents:
            return 0.0

        total_attempts = 0
        total_successes = 0

        try:
            for agent in self.agents:
                if not hasattr(agent, 'daily_stress_events'):
                    continue

                daily_events = agent.daily_stress_events
                if not isinstance(daily_events, list):
                    continue

                for event in daily_events:
                    if not isinstance(event, dict):
                        continue

                    if 'coped_successfully' in event:
                        total_attempts += 1
                        if event['coped_successfully']:
                            total_successes += 1

            if total_attempts == 0:
                return 0.0

            return total_successes / total_attempts
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _get_avg_challenge(self) -> float:
        """Get average challenge from stress events."""
        stress_events_data = self._collect_stress_events_data()
        return stress_events_data.get('avg_challenge', 0.0)

    def _get_avg_hindrance(self) -> float:
        """Get average hindrance from stress events."""
        stress_events_data = self._collect_stress_events_data()
        return stress_events_data.get('avg_hindrance', 0.0)

    def _get_challenge_hindrance_ratio(self) -> float:
        """Get challenge-hindrance ratio."""
        challenge_hindrance_data = self._collect_challenge_hindrance_data()
        return challenge_hindrance_data.get('ratio', 0.0)

    def _get_avg_consecutive_hindrances(self) -> float:
        """Get average consecutive hindrances."""
        consecutive_hindrances_data = self._collect_consecutive_hindrances_data()
        return consecutive_hindrances_data.get('avg', 0.0)

    def get_agent_time_series_data(self) -> pd.DataFrame:
        """
        Get time series data for agent-level metrics.

        Returns:
            DataFrame with agent-level time series data
        """
        if not hasattr(self, 'datacollector') or self.datacollector is None:
            return pd.DataFrame()

        # Get agent data from DataCollector
        agent_data = self.datacollector.get_agent_vars_dataframe()
        if agent_data.empty:
            return pd.DataFrame()

        return agent_data

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        """
        Get model variables dataframe from DataCollector.

        Returns:
            DataFrame with model variables
        """
        if not hasattr(self, 'datacollector') or self.datacollector is None:
            return pd.DataFrame()

        return self.datacollector.get_model_vars_dataframe()

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        """
        Get agent variables dataframe from DataCollector.

        Returns:
            DataFrame with agent variables
        """
        if not hasattr(self, 'datacollector') or self.datacollector is None:
            return pd.DataFrame()

        return self.datacollector.get_agent_vars_dataframe()


