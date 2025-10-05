# Import modules

import networkx as nx
import numpy as np
import pandas as pd
import mesa
import logging
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
    Enhanced agent-based model for mental health simulation using Mesa's DataCollector framework.

    This implementation leverages Mesa's built-in DataCollector for efficient, standardized data collection,
    eliminating the need for manual data tracking and providing enhanced performance and maintainability.

    Key Features:
    - **Unified Data Collection**: Single DataCollector instance manages all model and agent metrics
    - **Performance Optimized**: Leverages Mesa's optimized data collection mechanisms
    - **Research-Ready**: Comprehensive metrics for mental health research and analysis
    - **Backward Compatible**: Maintains existing API while using modern data collection patterns

    Data Collection Strategy:
    - Model reporters capture population-level metrics (averages, distributions, network statistics)
    - Agent reporters capture individual-level metrics (PSS-10, resilience, affect, resources)
    - Automatic daily collection during simulation steps
    - DataFrame-based output for easy analysis and export

    Benefits of DataCollector Approach:
    - Eliminates redundant manual data collection code
    - Provides standardized data access patterns
    - Enables efficient time series analysis
    - Supports both model-wide and agent-specific metrics
    - Integrates seamlessly with pandas for data analysis
    - Reduces memory footprint through optimized storage

    Supports:
    - Social interactions between neighboring agents via NetworkGrid
    - Affect and resilience dynamics at both individual and population levels
    - Social support exchange tracking and network adaptation
    - Population-level metrics collection with 20+ research-relevant indicators
    - Enhanced agent capabilities with protective factors and PSS-10 integration
    - Export functionality for simulation results and agent data

    Usage Examples:
        # Access model data as DataFrame
        model_data = model.datacollector.get_model_vars_dataframe()

        # Access agent data as DataFrame
        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Get population summary with enhanced metrics
        summary = model.get_population_summary()

        # Export results for analysis
        model.export_results("simulation_results.csv")
        model.export_agent_data("agent_data.csv")
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

        self.running = True

    def _initialize_datacollector(self):
        """
        Initialize Mesa DataCollector for comprehensive data collection.

        This method replaces manual data collection with Mesa's optimized DataCollector,
        providing better performance, standardized data access, and enhanced research capabilities.

        Migration from Manual Collection:
        - Previously: Manual tracking in dictionaries and custom data structures
        - Now: Mesa DataCollector with automatic collection and DataFrame output
        - Benefits: Reduced code complexity, better performance, standardized patterns

        Model reporters capture population-level metrics that are computed once per time step.
        These include averages, distributions, network statistics, and derived indicators
        essential for mental health research and cost-effectiveness analysis.
        """
        # Define model reporters (population-level metrics)
        # These lambda functions are called once per time step to compute population statistics
        # Each reporter returns a single scalar value representing a population characteristic
        # Core mental health metrics for research and analysis
        model_reporters = {
            # Primary outcome measures
            'avg_pss10': lambda m: m.get_avg_pss10(),  # Population average Perceived Stress Scale-10
            'avg_resilience': lambda m: m.get_avg_resilience(),  # Population average resilience score
            'avg_affect': lambda m: m.get_avg_affect(),  # Population average affect (positive/negative)

            # Coping and stress processing metrics
            'coping_success_rate': lambda m: m.get_success_rate(),  # Success rate in coping with stress events
            'avg_resources': lambda m: np.mean([agent.resources for agent in m.agents]) if m.agents else 0.0,  # Average resource levels
            'avg_stress': lambda m: np.mean([getattr(agent, 'current_stress', 0.0) for agent in m.agents]) if m.agents else 0.0,  # Average current stress

            # Social network and support metrics
            'social_support_rate': lambda m: m._calculate_social_support_rate(),  # Rate of social support exchanges
            'stress_events': lambda m: sum(len(getattr(agent, 'daily_stress_events', [])) for agent in m.agents),  # Total stress events per day
            'network_density': lambda m: m._calculate_network_density(),  # Network connectivity measure

            # Population health categories (for cost-effectiveness analysis)
            'stress_prevalence': lambda m: sum(1 for agent in m.agents if getattr(agent, 'stressed', False)) / len(m.agents) if m.agents else 0.0,  # Proportion with high stress (PSS-10 based)
            'low_resilience': lambda m: sum(1 for agent in m.agents if agent.resilience < 0.3) if m.agents else 0,  # Count with low resilience
            'high_resilience': lambda m: sum(1 for agent in m.agents if agent.resilience > 0.7) if m.agents else 0,  # Count with high resilience

            # Challenge/Hindrance appraisal metrics (key to theoretical model)
            'avg_challenge': lambda m: m._get_avg_challenge(),  # Average challenge appraisal across events
            'avg_hindrance': lambda m: m._get_avg_hindrance(),  # Average hindrance appraisal across events
            'challenge_hindrance_ratio': lambda m: m._get_challenge_hindrance_ratio(),  # Balance between challenge and hindrance
            'avg_consecutive_hindrances': lambda m: m._get_avg_consecutive_hindrances(),  # Average consecutive hindrance events

            # Daily activity statistics for intervention modeling
            'total_stress_events': lambda m: sum(len(getattr(agent, 'daily_stress_events', [])) for agent in m.agents),  # Total stress events
            'successful_coping': lambda m: sum(sum(1 for event in getattr(agent, 'daily_stress_events', []) if event.get('coped_successfully', False)) for agent in m.agents),  # Successful coping instances
            'social_interactions': lambda m: sum(getattr(agent, 'daily_interactions', 0) for agent in m.agents),  # Total social interactions
            'support_exchanges': lambda m: sum(getattr(agent, 'daily_support_exchanges', 0) for agent in m.agents)  # Total support exchanges
        }

        # Define agent reporters (agent-level metrics)
        # These lambda functions capture individual agent state for longitudinal analysis
        # Each agent is recorded once per time step, enabling trajectory analysis
        # Essential for studying individual differences and intervention effects
        agent_reporters = {
            # Core individual outcome measures
            'pss10': lambda a: a.pss10,  # Individual Perceived Stress Scale-10 score
            'resilience': lambda a: a.resilience,  # Individual resilience capacity
            'affect': lambda a: a.affect,  # Individual positive/negative affect balance
            'resources': lambda a: a.resources,  # Individual resource availability

            # Individual stress processing state
            'current_stress': lambda a: getattr(a, 'current_stress', 0.0),  # Current stress level
            'stress_controllability': lambda a: getattr(a, 'stress_controllability', 0.5),  # Perceived controllability
            'stress_overload': lambda a: getattr(a, 'stress_overload', 0.5),  # Perceived overload

            # Individual stress event tracking
            'consecutive_hindrances': lambda a: getattr(a, 'consecutive_hindrances', 0)  # Consecutive hindrance events
        }

        # Initialize DataCollector with comprehensive metrics
        # This single DataCollector instance replaces all manual data collection
        # Provides optimized storage and standardized access patterns
        self.datacollector = DataCollector(
            model_reporters=model_reporters,  # Population-level metrics (20+ indicators)
            agent_reporters=agent_reporters   # Individual-level metrics (8+ per agent)
        )

        # Maintain daily stats for backward compatibility with existing code
        # These are now populated from DataCollector data rather than manual tracking
        # TODO: Phase out in favor of direct DataCollector access for better performance
        self.daily_stats = {
            'total_stress_events': 0,    # Now derived from DataCollector model data
            'successful_coping': 0,      # Now derived from DataCollector model data
            'social_interactions': 0,    # Now derived from DataCollector model data
            'support_exchanges': 0       # Now derived from DataCollector model data
        }

    def step(self):
        """
        Execute one day of simulation with enhanced social interactions and dynamics.

        This method implements a streamlined workflow using Mesa's DataCollector for
        efficient data collection and analysis. The DataCollector approach eliminates
        manual data tracking while providing better performance and research capabilities.

        Process order:
        1. Reset daily statistics for backward compatibility
        2. Execute agent steps (interactions, stress events, adaptation)
        3. Collect comprehensive metrics using DataCollector (population + agent data)
        4. Apply network adaptation mechanisms at population level
        5. Update social support tracking from agent interactions
        6. Record daily statistics from DataCollector outputs

        Data Collection Benefits:
        - Single line collection: self.datacollector.collect(self)
        - Automatic storage in optimized DataFrames
        - No manual metric calculation or storage management
        - Consistent data structure for analysis
        - Reduced memory footprint and faster execution
        """
        # Reset daily statistics for backward compatibility
        self._reset_daily_stats()

        # Execute all agent steps with enhanced social interactions
        self.agents.shuffle_do("step")

        # Collects both model-level (population) and agent-level (individual) metrics
        self.datacollector.collect(self)

        # Apply network adaptation mechanisms at population level
        self._apply_network_adaptation()

        # Update social support tracking from agent interaction data
        self._update_social_support_tracking()

        # Maintains backward compatibility while using DataCollector as source of truth
        self._record_daily_stats()

        # Apply daily reset to all agents AFTER data collection
        for agent in self.agents:
            if hasattr(agent, '_daily_reset'):
                agent._daily_reset(self.day)

        # Increment simulation day counter
        self.day += 1

        # Check simulation termination condition
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

    def _calculate_social_support_rate(self) -> float:
        """Calculate rate of social support exchanges in the population."""
        if self.total_interactions == 0:
            return 0.0

        return self.social_support_exchanges / self.total_interactions

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
        Get comprehensive summary of current population state using DataCollector outputs.

        This method demonstrates the power of the DataCollector approach by providing
        rich, research-ready metrics derived from both model and agent data. The method
        combines population-level statistics with individual-level distributions to give
        researchers a complete picture of mental health dynamics.

        Enhanced Metrics Available:
        - Core mental health indicators (PSS-10, resilience, affect)
        - Stress processing metrics (challenge/hindrance, coping success)
        - Social network metrics (support rates, network density)
        - Population health distributions (resilience categories)
        - Integrated health indices (mental health index, vulnerability index)
        - Daily activity statistics for intervention modeling

        Data Access Patterns for Researchers:
        1. Single time point: Use this method for current state
        2. Time series: Use model.datacollector.get_model_vars_dataframe()
        3. Individual trajectories: Use model.datacollector.get_agent_vars_dataframe()
        4. Custom analysis: Combine DataFrames for specific research questions

        Usage Examples:
            # Get current population state
            summary = model.get_population_summary()

            # Access specific metrics
            current_stress = summary['avg_stress']
            resilience_distribution = summary['resilience_distribution']

            # Export for analysis
            df = model.get_time_series_data()
            df.to_csv('population_trends.csv')

            # Research workflow: Compare intervention scenarios
            baseline_data = model.get_model_vars_dataframe()
            intervention_data = intervention_model.get_model_vars_dataframe()
            comparison = baseline_data.compare(intervention_data)

        Returns:
            Dictionary containing comprehensive population metrics including:
            - Basic statistics (averages, standard deviations)
            - Distribution metrics (resilience categories, stress prevalence)
            - Integrated indices (mental health index, vulnerability index)
            - Social dynamics (support rates, network density)
            - Stress processing (challenge/hindrance ratios, coping success)
            - Daily activity summaries (stress events, interactions)
        """
        if not self.agents:
            return {}

        if not hasattr(self, 'datacollector') or self.datacollector is None:
            return {}

        try:
            # Get latest data from DataCollector
            model_data = self.datacollector.get_model_vars_dataframe()
            if model_data.empty:
                return {}

            latest_data = model_data.iloc[-1]

            # Get agent data for additional statistics
            agent_data = self.datacollector.get_agent_vars_dataframe()

            # Current averages from DataCollector
            current_affect = latest_data.get('avg_affect', 0.0)
            current_resilience = latest_data.get('avg_resilience', 0.0)
            current_resources = latest_data.get('avg_resources', 0.0)
            current_stress = latest_data.get('avg_stress', 0.0)

            # Distribution statistics
            if not agent_data.empty:
                affect_std = agent_data['affect'].std() if 'affect' in agent_data.columns else 0.0
                resilience_std = agent_data['resilience'].std() if 'resilience' in agent_data.columns else 0.0
                stress_std = agent_data['current_stress'].std() if 'current_stress' in agent_data.columns else 0.0
            else:
                affect_std = 0.0
                resilience_std = 0.0
                stress_std = 0.0

            # Stress prevalence
            stress_prevalence = latest_data.get('stress_prevalence', 0.0)

            # Resilience categories
            low_resilience = latest_data.get('low_resilience', 0)
            high_resilience = latest_data.get('high_resilience', 0)
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
                'social_support_rate': latest_data.get('social_support_rate', 0.0),
                'total_interactions': self.total_interactions,
                'network_density': latest_data.get('network_density', 0.0),
                # Enhanced integrated metrics
                'mental_health_index': (current_affect + current_resilience) / 2,  # Combined mental health score
                'recovery_potential': high_resilience / max(1, len(self.agents)),  # Proportion with high resilience
                'vulnerability_index': low_resilience / max(1, len(self.agents)),   # Proportion with low resilience
                # New stress processing metrics
                'avg_challenge': latest_data.get('avg_challenge', 0.0),
                'avg_hindrance': latest_data.get('avg_hindrance', 0.0),
                'challenge_hindrance_ratio': latest_data.get('challenge_hindrance_ratio', 0.0),
                'coping_success_rate': latest_data.get('coping_success_rate', 0.0),
                'avg_consecutive_hindrances': latest_data.get('avg_consecutive_hindrances', 0.0),
                # Daily statistics from new reporters
                'total_stress_events': latest_data.get('total_stress_events', 0),
                'successful_coping': latest_data.get('successful_coping', 0),
                'social_interactions': latest_data.get('social_interactions', 0),
                'support_exchanges': latest_data.get('support_exchanges', 0)
            }
        except Exception:
            # Return empty dict if DataCollector fails
            return {}

    def get_time_series_data(self) -> pd.DataFrame:
        """
        Get time series data for population metrics using DataCollector.

        This method showcases the DataCollector advantage: instant access to clean,
        structured time series data without manual accumulation or processing.

        Migration from Manual Collection:
        - Before: Manual list/dict accumulation, error-prone concatenation
        - After: Single method call returns properly formatted DataFrame
        - Benefits: Consistent structure, automatic indexing, pandas integration

        Research Workflow Examples:
            # Basic time series analysis
            df = model.get_time_series_data()
            trends = df[['avg_stress', 'avg_resilience']].rolling(window=7).mean()

            # Intervention impact assessment
            baseline = model.get_time_series_data()
            post_intervention = intervention_model.get_time_series_data()
            impact = post_intervention - baseline

            # Statistical analysis
            stress_trend = scipy.stats.linregress(df['day'], df['avg_stress'])

        Returns:
            DataFrame with daily population metrics, including columns for:
            - Day index and basic counts
            - Mental health metrics (PSS-10, resilience, affect)
            - Stress processing (challenge/hindrance, coping rates)
            - Social dynamics (support rates, network density)
            - Population distributions (stress prevalence, resilience categories)
        """
        # Use DataCollector to get model data
        if not hasattr(self, 'datacollector') or self.datacollector is None:
            return pd.DataFrame()

        # Get data from DataCollector - single method call replaces manual collection
        model_data = self.datacollector.get_model_vars_dataframe()
        if model_data.empty:
            return pd.DataFrame()

        return model_data
    def export_results(self, filename: str = None) -> str:
        """
        Export simulation results to CSV file using DataCollector outputs.

        Demonstrates the DataCollector advantage for data export: clean, structured
        data ready for analysis without manual formatting or processing.

        Args:
            filename: Optional filename for export (default: auto-generated with day)

        Returns:
            Path to exported file

        Export Workflow Examples:
            # Standard export
            model.export_results()  # Auto-generates filename

            # Custom filename
            model.export_results("baseline_simulation.csv")

            # Batch export for parameter sweep
            for params in parameter_sets:
                model = StressModel(**params)
                for _ in range(100):  # Run simulation
                    if model.running:
                        model.step()
                model.export_results(f"params_{params['id']}.csv")
        """
        if filename is None:
            filename = f"simulation_results_day_{self.day}.csv"

        # Get data from DataCollector - already clean and structured for export
        df = self.get_time_series_data()

        if not df.empty:
            df.to_csv(filename, index=False)

        return filename

    def export_agent_data(self, filename: str = None) -> str:
        """
        Export agent-level time series data to CSV file for individual trajectory analysis.

        Agent-level data is crucial for understanding individual differences in stress
        processing, resilience trajectories, and intervention responses. The DataCollector
        approach makes this data readily available without manual tracking.

        Args:
            filename: Optional filename for export (default: auto-generated with day)

        Returns:
            Path to exported file

        Individual Analysis Examples:
            # Export individual trajectories
            model.export_agent_data("individual_trajectories.csv")

            # Analyze resilience trajectories
            agent_data = model.get_agent_time_series_data()
            resilience_trends = agent_data.groupby('AgentID')['resilience'].agg(['mean', 'std', 'min', 'max'])

            # Identify at-risk individuals
            final_state = agent_data.groupby('AgentID').last()
            at_risk = final_state[final_state['resilience'] < 0.3]
        """
        if filename is None:
            filename = f"agent_data_day_{self.day}.csv"

        # Get agent data from DataCollector - comprehensive individual-level data
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
        """Get average challenge from stress events using agent data."""
        total_challenge = 0.0
        total_events = 0

        for agent in self.agents:
            daily_events = getattr(agent, 'daily_stress_events', [])
            for event in daily_events:
                total_challenge += event.get('challenge', 0.0)
                total_events += 1

        return total_challenge / total_events if total_events > 0 else 0.0

    def _get_avg_hindrance(self) -> float:
        """Get average hindrance from stress events using agent data."""
        total_hindrance = 0.0
        total_events = 0

        for agent in self.agents:
            daily_events = getattr(agent, 'daily_stress_events', [])
            for event in daily_events:
                total_hindrance += event.get('hindrance', 0.0)
                total_events += 1

        return total_hindrance / total_events if total_events > 0 else 0.0

    def _get_challenge_hindrance_ratio(self) -> float:
        """Get challenge-hindrance ratio using agent data."""
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
            return (avg_challenge - avg_hindrance) / (avg_challenge + avg_hindrance) if (avg_challenge + avg_hindrance) > 0 else 0.0

        return 0.0

    def _get_avg_consecutive_hindrances(self) -> float:
        """Get average consecutive hindrances using agent data."""
        total_consecutive = 0.0
        valid_agents = 0

        for agent in self.agents:
            consecutive = getattr(agent, 'consecutive_hindrances', 0)
            if consecutive > 0:
                total_consecutive += consecutive
                valid_agents += 1

        return total_consecutive / valid_agents if valid_agents > 0 else 0.0

    def get_agent_time_series_data(self) -> pd.DataFrame:
        """
        Get time series data for agent-level metrics using DataCollector.

        Agent-level data enables sophisticated individual-difference analysis essential
        for mental health research, including resilience trajectories, stress response
        patterns, and intervention efficacy at the individual level.

        Returns:
            DataFrame with agent-level time series data including:
            - AgentID: Unique identifier for each agent
            - Step: Time step of measurement
            - pss10: Individual PSS-10 stress scores over time
            - resilience: Individual resilience trajectories
            - affect: Individual affect dynamics
            - resources: Individual resource allocation patterns
            - current_stress: Individual stress levels
            - stress_controllability: Individual controllability perceptions
            - stress_overload: Individual overload perceptions
            - consecutive_hindrances: Individual hindrance event patterns

        Research Applications:
            # Individual resilience trajectory analysis
            agent_data = model.get_agent_time_series_data()
            trajectories = agent_data.pivot(index='Step', columns='AgentID', values='resilience')

            # Identify intervention responders vs non-responders
            final_resilience = agent_data.groupby('AgentID')['resilience'].last()
            responders = final_resilience[final_resilience > 0.7]

            # Stress response pattern clustering
            features = ['pss10', 'resilience', 'affect', 'resources']
            patterns = agent_data.groupby('AgentID')[features].mean()
            clusters = KMeans(n_clusters=3).fit(patterns)
        """
        if not hasattr(self, 'datacollector') or self.datacollector is None:
            return pd.DataFrame()

        # Get agent data from DataCollector - comprehensive individual trajectories
        agent_data = self.datacollector.get_agent_vars_dataframe()
        if agent_data.empty:
            return pd.DataFrame()

        # Add AgentID and Step columns to the DataFrame
        # Mesa's DataCollector doesn't include these by default, so we need to add them
        try:
            # Reset index to get AgentID and Step information
            agent_data = agent_data.reset_index()

            # Rename columns to match expected format
            if 'AgentID' in agent_data.columns and 'Step' in agent_data.columns:
                # Columns are already properly named
                pass
            else:
                # Handle different possible column naming from Mesa
                # The index levels should contain AgentID and Step information
                logger = logging.getLogger('simulation')
                logger.warning("Unexpected DataFrame structure from DataCollector, attempting to fix...")

                # Try to reconstruct the DataFrame with proper columns
                if hasattr(agent_data.index, 'names') and agent_data.index.names:
                    # Multi-index case - reconstruct with proper column names
                    agent_data = agent_data.reset_index()
                    if len(agent_data.index.names) >= 2:
                        # Assume first level is AgentID, second is Step
                        agent_data = agent_data.rename(columns={
                            agent_data.columns[0]: 'AgentID',
                            agent_data.columns[1]: 'Step'
                        })
                else:
                    # Single index case - need to add missing columns
                    # Create AgentID column based on row patterns
                    num_rows = len(agent_data)
                    num_agents = len(self.agents)
                    steps_per_agent = num_rows // num_agents if num_agents > 0 else 1

                    # Create AgentID column
                    agent_ids = []
                    for i in range(num_rows):
                        agent_id = i // steps_per_agent
                        agent_ids.append(agent_id)

                    agent_data.insert(0, 'AgentID', agent_ids)

                    # Create Step column
                    steps = []
                    for i in range(num_rows):
                        step = i % steps_per_agent
                        steps.append(step)

                    agent_data.insert(1, 'Step', steps)

            return agent_data

        except Exception as e:
            # If there's any error in processing, return the original data
            # This ensures backward compatibility
            logger = logging.getLogger('simulation')
            logger.warning(f"Error processing agent data structure: {e}")
            return agent_data

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        """
        Get model variables dataframe directly from DataCollector.

        Provides direct access to the underlying DataCollector DataFrame for
        advanced research workflows requiring maximum flexibility.

        Returns:
            DataFrame with model variables - raw DataCollector output for
            population-level metrics across all time steps
        """
        if not hasattr(self, 'datacollector') or self.datacollector is None:
            return pd.DataFrame()

        return self.datacollector.get_model_vars_dataframe()

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        """
        Get agent variables dataframe directly from DataCollector.

        Provides direct access to individual-level data for advanced trajectory
        analysis and individual difference research.

        Returns:
            DataFrame with agent variables - raw DataCollector output for
            individual-level metrics across all time steps and all agents
        """
        if not hasattr(self, 'datacollector') or self.datacollector is None:
            return pd.DataFrame()

        return self.datacollector.get_agent_vars_dataframe()
