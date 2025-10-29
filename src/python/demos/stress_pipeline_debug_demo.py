"""
Stress Processing Pipeline Debug Demo

This demo script comprehensively debugs the stress processing pipeline and PSS-10 correlation issues.
It focuses on identifying where expected negative correlations between resilience/affect and PSS-10
are breaking down in the stress processing pipeline.

The demo provides detailed debugging output showing:
- Resilience vs coping success rates over time
- Stress level changes vs PSS-10 updates
- Coping probability distributions by resilience levels
- Step-by-step correlation evolution
- Validation of theoretical expectations

This will help identify exactly where in the stress processing pipeline the expected correlations
between resilience/affect and PSS-10 are breaking down.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append('.')

from src.python.model import StressModel
from src.python.agent import Person
from src.python.config import get_config
from src.python.stress_utils import (
    generate_stress_event, apply_weights, process_stress_event,
    generate_pss10_from_stress_dimensions, update_stress_dimensions_from_pss10_feedback,
    update_stress_dimensions_from_event, validate_theoretical_correlations,
    StressEvent, AppraisalWeights, ThresholdParams
)
from src.python.affect_utils import (
    compute_coping_probability, determine_coping_outcome_and_psychological_impact,
    StressProcessingConfig, get_neighbor_affects
)


class StressPipelineDebugger:
    """
    Comprehensive debugger for the stress processing pipeline and PSS-10 correlations.

    Tracks detailed data throughout the simulation to identify where correlations
    between resilience/affect and PSS-10 break down.
    """

    def __init__(self, num_agents: int = 20, max_steps: int = 50, seed: int = 42):
        """
        Initialize the stress pipeline debugger.

        Args:
            num_agents: Number of agents in the simulation
            max_steps: Number of simulation steps to run
            seed: Random seed for reproducible results
        """
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.seed = seed

        # Create model with diverse agent population for correlation analysis
        self.model = StressModel(N=num_agents, max_days=max_steps, seed=seed)

        # Storage for debugging data
        self.debug_data = {
            'step': [],
            'agent_id': [],
            'resilience': [],
            'affect': [],
            'current_stress': [],
            'pss10_score': [],
            'stress_controllability': [],
            'stress_overload': [],
            'coping_success': [],
            'challenge': [],
            'hindrance': [],
            'social_influence': [],
            'coping_probability': [],
            'stress_level_change': [],
            'pss10_change': []
        }

        # Track previous values for change calculations
        self.previous_values = {}

        # Correlation tracking over time
        self.correlation_history = {
            'step': [],
            'resilience_pss10_corr': [],
            'affect_pss10_corr': [],
            'resilience_coping_corr': [],
            'stress_pss10_corr': []
        }

        # Step-by-step metrics logging
        self.step_metrics_log = []

    def _force_debug_stress_events(self):
        """Force stress events for debugging purposes to ensure we have data to analyze."""
        print("  Forcing debug stress events for all agents...")

        for agent in self.model.agents:
            # Create diverse stress events for each agent
            if agent.unique_id % 3 == 0:
                # Force high challenge event (should improve coping for high resilience agents)
                event = StressEvent(controllability=0.8, overload=0.2)
                challenge, hindrance = 0.9, 0.1
            elif agent.unique_id % 3 == 1:
                # Force high hindrance event (should worsen outcomes)
                event = StressEvent(controllability=0.2, overload=0.8)
                challenge, hindrance = 0.1, 0.9
            else:
                # Force balanced event
                event = StressEvent(controllability=0.5, overload=0.5)
                challenge, hindrance = 0.5, 0.5

            # Get neighbor affects for social influence
            neighbor_affects = get_neighbor_affects(agent, self.model)

            # Process the stress event
            stress_config = StressProcessingConfig()
            new_affect, new_resilience, new_stress, coped_successfully = determine_coping_outcome_and_psychological_impact(
                current_affect=agent.affect,
                current_resilience=agent.resilience,
                current_stress=agent.current_stress,
                challenge=challenge,
                hindrance=hindrance,
                neighbor_affects=neighbor_affects,
                config=stress_config
            )

            # Update agent state
            agent.affect = new_affect
            agent.resilience = new_resilience
            agent.current_stress = new_stress

            # Update stress dimensions and PSS-10
            (
                agent.stress_controllability,
                agent.stress_overload,
                agent.recent_stress_intensity,
                agent.stress_momentum
            ) = update_stress_dimensions_from_event(
                current_controllability=agent.stress_controllability,
                current_overload=agent.stress_overload,
                challenge=challenge,
                hindrance=hindrance,
                coped_successfully=coped_successfully,
                is_stressful=True,
                volatility=agent.volatility
            )

            # Generate PSS-10 from updated stress dimensions
            pss10_data = generate_pss10_from_stress_dimensions(
                stress_controllability=agent.stress_controllability,
                stress_overload=agent.stress_overload,
                recent_stress_intensity=agent.recent_stress_intensity,
                stress_momentum=agent.stress_momentum,
                rng=agent._rng
            )
            agent.pss10_responses = pss10_data['pss10_responses']
            agent.pss10 = pss10_data['pss10_score']
            agent.stressed = pss10_data['stressed']

            # Manually add to daily stress events for tracking
            if not hasattr(agent, 'daily_stress_events'):
                agent.daily_stress_events = []
            agent.daily_stress_events.append({
                'challenge': challenge,
                'hindrance': hindrance,
                'is_stressed': True,
                'stress_level': new_stress,
                'coped_successfully': coped_successfully,
                'event_controllability': event.controllability,
                'event_overload': event.overload
            })

        print(f"  Forced {len(self.model.agents)} stress events for debugging")

    def run_debugging_simulation(self) -> Dict[str, Any]:
        """
        Run the extended debugging simulation with comprehensive data collection.

        Returns:
            Dictionary containing all debugging results and analysis
        """
        print("=" * 80)
        print("STRESS PROCESSING PIPELINE DEBUG DEMO")
        print("=" * 80)
        print(f"Running simulation with {self.num_agents} agents for {self.max_steps} steps")
        print(f"Seed: {self.seed}")
        print()

        # Initialize previous values tracking
        self._initialize_previous_values()

        for step in range(self.max_steps):
            if step % 10 == 0:
                print(f"Step {step + 1}/{self.max_steps}")

            # Execute one simulation step
            self.model.step()

            # Force some stress events for debugging if none are occurring naturally
            if step > 5 and step % 15 == 0:  # Every 15 steps after step 5
                self._force_debug_stress_events()
                step_description = "Forced Debug Events"
            else:
                step_description = ""

            # Log comprehensive metrics for this step
            self._calculate_and_log_step_metrics(step, step_description)

            # Debug: Check if any agents have stress events
            if step % 10 == 0:
                total_events = sum(len(getattr(agent, 'daily_stress_events', [])) for agent in self.model.agents)
                print(f"  Debug: Total stress events across all agents: {total_events}")

                # Show first agent's state for debugging
                if self.model.agents:
                    agent = self.model.agents[0]
                    print(f"  Debug: Agent 0 - Stress: {agent.current_stress:.3f}, PSS-10: {agent.pss10}, Events: {len(getattr(agent, 'daily_stress_events', []))}")

            # Collect detailed debugging data for each agent
            self._collect_step_data(step)

            # Analyze correlations every 5 steps
            if step % 5 == 0 and step > 0:
                self._analyze_step_correlations(step)

        # Perform final comprehensive analysis
        analysis_results = self._perform_comprehensive_analysis()

        return {
            'debug_data': self.debug_data,
            'correlation_history': self.correlation_history,
            'analysis_results': analysis_results,
            'model': self.model,
            'step_metrics_log': self.step_metrics_log
        }

    def _initialize_previous_values(self):
        """Initialize tracking of previous values for change calculations."""
        for i, agent in enumerate(self.model.agents):
            agent_id = agent.unique_id
            self.previous_values[agent_id] = {
                'current_stress': agent.current_stress,
                'pss10_score': agent.pss10
            }

    def _collect_step_data(self, step: int):
        """
        Collect detailed debugging data for the current step.

        Args:
            step: Current simulation step
        """
        for agent in self.model.agents:
            agent_id = agent.unique_id

            # Get neighbor affects for social influence calculation
            neighbor_affects = get_neighbor_affects(agent, self.model)

            # Get the most recent stress event if any occurred
            daily_events = getattr(agent, 'daily_stress_events', [])
            if daily_events:
                latest_event = daily_events[-1]

                # Calculate coping probability for this event
                coping_prob = compute_coping_probability(
                    latest_event['challenge'],
                    latest_event['hindrance'],
                    neighbor_affects,
                    StressProcessingConfig()
                )

                # Store comprehensive debugging data
                self.debug_data['step'].append(step + 1)
                self.debug_data['agent_id'].append(agent_id)
                self.debug_data['resilience'].append(agent.resilience)
                self.debug_data['affect'].append(agent.affect)
                self.debug_data['current_stress'].append(agent.current_stress)
                self.debug_data['pss10_score'].append(agent.pss10)
                self.debug_data['stress_controllability'].append(agent.stress_controllability)
                self.debug_data['stress_overload'].append(agent.stress_overload)
                self.debug_data['coping_success'].append(latest_event['coped_successfully'])
                self.debug_data['challenge'].append(latest_event['challenge'])
                self.debug_data['hindrance'].append(latest_event['hindrance'])
                self.debug_data['social_influence'].append(np.mean(neighbor_affects) if neighbor_affects else 0.0)
                self.debug_data['coping_probability'].append(coping_prob)

                # Calculate changes from previous step
                prev_stress = self.previous_values[agent_id]['current_stress']
                prev_pss10 = self.previous_values[agent_id]['pss10_score']

                self.debug_data['stress_level_change'].append(agent.current_stress - prev_stress)
                self.debug_data['pss10_change'].append(agent.pss10 - prev_pss10)

                # Update previous values
                self.previous_values[agent_id]['current_stress'] = agent.current_stress
                self.previous_values[agent_id]['pss10_score'] = agent.pss10

            # Also collect baseline data for all agents at regular intervals for correlation analysis
            if step % 5 == 0:  # Every 5 steps, collect baseline data from all agents
                self.debug_data['step'].append(step + 1)
                self.debug_data['agent_id'].append(agent_id)
                self.debug_data['resilience'].append(agent.resilience)
                self.debug_data['affect'].append(agent.affect)
                self.debug_data['current_stress'].append(agent.current_stress)
                self.debug_data['pss10_score'].append(agent.pss10)
                self.debug_data['stress_controllability'].append(agent.stress_controllability)
                self.debug_data['stress_overload'].append(agent.stress_overload)
                # For baseline data, use placeholder values for event-specific fields
                self.debug_data['coping_success'].append(0.5)  # Neutral value
                self.debug_data['challenge'].append(0.0)       # No event
                self.debug_data['hindrance'].append(0.0)      # No event
                self.debug_data['social_influence'].append(np.mean(neighbor_affects) if neighbor_affects else 0.0)
                self.debug_data['coping_probability'].append(0.5)  # Neutral value

                # Calculate changes from previous step
                prev_stress = self.previous_values[agent_id]['current_stress']
                prev_pss10 = self.previous_values[agent_id]['pss10_score']

                self.debug_data['stress_level_change'].append(agent.current_stress - prev_stress)
                self.debug_data['pss10_change'].append(agent.pss10 - prev_pss10)

                # Update previous values
                self.previous_values[agent_id]['current_stress'] = agent.current_stress
                self.previous_values[agent_id]['pss10_score'] = agent.pss10

    def _analyze_step_correlations(self, step: int):
        """
        Analyze correlations at the current step using only data from actual stress events.

        Filters out baseline data (collected every 5 steps) that contains neutral values
        and dilutes correlations. Only includes entries where actual stress events occurred
        (challenge > 0.0 or coping_success != 0.5).

        Args:
            step: Current simulation step
        """
        if len(self.debug_data['resilience']) < 5:
            return  # Need minimum data for correlations

        # Filter data to only include actual stress events
        # Exclude baseline data where challenge=0.0 and coping_success=0.5 (neutral values)
        event_indices = [
            i for i in range(len(self.debug_data['challenge']))
            if self.debug_data['challenge'][i] > 0.0 and self.debug_data['coping_success'][i] != 0.5
        ]

        if len(event_indices) < 5:
            return  # Need sufficient event data for meaningful correlations

        # Extract only event data using filtered indices
        resilience_vals = [self.debug_data['resilience'][i] for i in event_indices]
        affect_vals = [self.debug_data['affect'][i] for i in event_indices]
        pss10_vals = [self.debug_data['pss10_score'][i] for i in event_indices]
        coping_vals = [self.debug_data['coping_success'][i] for i in event_indices]
        stress_vals = [self.debug_data['current_stress'][i] for i in event_indices]

        # Calculate correlations using only event data
        try:
            resilience_pss10_corr = np.corrcoef(resilience_vals, pss10_vals)[0, 1]
            affect_pss10_corr = np.corrcoef(affect_vals, pss10_vals)[0, 1]
            resilience_coping_corr = np.corrcoef(resilience_vals, coping_vals)[0, 1]
            stress_pss10_corr = np.corrcoef(stress_vals, pss10_vals)[0, 1]

            # Store correlation history
            self.correlation_history['step'].append(step + 1)
            self.correlation_history['resilience_pss10_corr'].append(resilience_pss10_corr)
            self.correlation_history['affect_pss10_corr'].append(affect_pss10_corr)
            self.correlation_history['resilience_coping_corr'].append(resilience_coping_corr)
            self.correlation_history['stress_pss10_corr'].append(stress_pss10_corr)

            # Debug output with event count
            if step % 10 == 0:
                print(f"  Debug: Step {step + 1} correlations (events={len(event_indices)}) - Resilience-PSS10: {resilience_pss10_corr:.4f}, Affect-PSS10: {affect_pss10_corr:.4f}")

        except (IndexError, ValueError) as e:
            # Handle cases where correlation calculation fails
            print(f"  Debug: Correlation calculation failed at step {step + 1}: {e}")
            pass

    def _perform_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the debugging data.

        Returns:
            Dictionary containing analysis results
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE STRESS PROCESSING PIPELINE ANALYSIS")
        print("=" * 80)

        analysis = {}

        # 1. Resilience vs Coping Success Analysis
        print("\n1. RESILIENCE VS COPING SUCCESS ANALYSIS")
        print("-" * 50)
        analysis['resilience_coping'] = self._analyze_resilience_coping_relationship()

        # 2. PSS-10 vs Stress Reflection Analysis
        print("\n2. PSS-10 VS STRESS REFLECTION ANALYSIS")
        print("-" * 50)
        analysis['pss10_stress'] = self._analyze_pss10_stress_reflection()

        # 3. Coping Probability Mechanism Analysis
        print("\n3. COPING PROBABILITY MECHANISM ANALYSIS")
        print("-" * 50)
        analysis['coping_probability'] = self._analyze_coping_probability_mechanism()

        # 4. Correlation Evolution Analysis
        print("\n4. CORRELATION EVOLUTION ANALYSIS")
        print("-" * 50)
        analysis['correlation_evolution'] = self._analyze_correlation_evolution()

        # 5. Stress Processing Pipeline Validation
        print("\n5. STRESS PROCESSING PIPELINE VALIDATION")
        print("-" * 50)
        analysis['pipeline_validation'] = self._validate_stress_processing_pipeline()

        return analysis

    def _analyze_resilience_coping_relationship(self) -> Dict[str, Any]:
        """Analyze the relationship between resilience and coping success."""
        resilience_vals = self.debug_data['resilience']
        coping_vals = self.debug_data['coping_success']

        if len(resilience_vals) == 0:
            return {'error': 'No data available'}

        # Calculate coping success rates by resilience quartiles
        data = list(zip(resilience_vals, coping_vals))
        df = pd.DataFrame(data, columns=['resilience', 'coping_success'])

        # Create resilience groups (handle low variation case)
        unique_resilience = df['resilience'].unique()
        if len(unique_resilience) <= 3:
            # Low variation case - use simple binary grouping
            median_resilience = df['resilience'].median()
            df['resilience_group'] = ['Low' if r <= median_resilience else 'High' for r in df['resilience']]

            quartile_stats = df.groupby('resilience_group').agg({
                'coping_success': ['mean', 'count', 'std'],
                'resilience': ['mean', 'min', 'max']
            }).round(4)
        else:
            # Sufficient variation for quartiles
            try:
                df['resilience_quartile'] = pd.qcut(df['resilience'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

                quartile_stats = df.groupby('resilience_quartile').agg({
                    'coping_success': ['mean', 'count', 'std'],
                    'resilience': ['mean', 'min', 'max']
                }).round(4)
            except ValueError:
                # Fallback to binary grouping if qcut still fails
                median_resilience = df['resilience'].median()
                df['resilience_group'] = ['Low' if r <= median_resilience else 'High' for r in df['resilience']]

                quartile_stats = df.groupby('resilience_group').agg({
                    'coping_success': ['mean', 'count', 'std'],
                    'resilience': ['mean', 'min', 'max']
                }).round(4)

        # Overall correlation
        correlation = np.corrcoef(resilience_vals, coping_vals)[0, 1]

        print(f"Overall Resilience-Coping Correlation: {correlation:.4f}")
        print(f"Sample size: {len(resilience_vals)} events")
        print("\nCoping Success Rates by Resilience Quartiles:")
        print(quartile_stats)

        return {
            'correlation': correlation,
            'quartile_stats': quartile_stats.to_dict(),
            'sample_size': len(resilience_vals)
        }

    def _analyze_pss10_stress_reflection(self) -> Dict[str, Any]:
        """Analyze how well PSS-10 scores reflect stress level changes."""
        pss10_vals = self.debug_data['pss10_score']
        stress_vals = self.debug_data['current_stress']
        stress_controllability = self.debug_data['stress_controllability']
        stress_overload = self.debug_data['stress_overload']

        if len(pss10_vals) == 0:
            return {'error': 'No data available'}

        # Calculate correlations
        pss10_stress_corr = np.corrcoef(pss10_vals, stress_vals)[0, 1]
        pss10_controllability_corr = np.corrcoef(pss10_vals, stress_controllability)[0, 1]
        pss10_overload_corr = np.corrcoef(pss10_vals, stress_overload)[0, 1]

        # Analyze PSS-10 responsiveness to stress changes
        stress_changes = self.debug_data['stress_level_change']
        pss10_changes = self.debug_data['pss10_change']

        # Calculate change correlation (how well PSS-10 changes track stress changes)
        change_correlation = np.corrcoef(stress_changes, pss10_changes)[0, 1]

        print(f"PSS-10 vs Current Stress Correlation: {pss10_stress_corr:.4f}")
        print(f"PSS-10 vs Stress Controllability Correlation: {pss10_controllability_corr:.4f}")
        print(f"PSS-10 vs Stress Overload Correlation: {pss10_overload_corr:.4f}")
        print(f"PSS-10 vs Stress Change Correlation: {change_correlation:.4f}")

        # Analyze PSS-10 responsiveness (how quickly it responds to stress changes)
        significant_stress_changes = [i for i, change in enumerate(stress_changes) if abs(change) > 0.1]
        if significant_stress_changes:
            avg_pss10_response = np.mean([abs(pss10_changes[i]) for i in significant_stress_changes])
            print(f"Average PSS-10 response to significant stress changes: {avg_pss10_response:.4f}")

        return {
            'pss10_stress_corr': pss10_stress_corr,
            'pss10_controllability_corr': pss10_controllability_corr,
            'pss10_overload_corr': pss10_overload_corr,
            'change_correlation': change_correlation,
            'avg_pss10_response': avg_pss10_response if significant_stress_changes else 0.0
        }

    def _analyze_coping_probability_mechanism(self) -> Dict[str, Any]:
        """Analyze the coping probability mechanism and its relationship to resilience."""
        resilience_vals = self.debug_data['resilience']
        coping_probs = self.debug_data['coping_probability']
        coping_success = self.debug_data['coping_success']

        if len(resilience_vals) == 0:
            return {'error': 'No data available'}

        # Calculate correlation between resilience and coping probability
        prob_correlation = np.corrcoef(resilience_vals, coping_probs)[0, 1]

        # Calculate actual coping success vs predicted probability
        success_rate_by_prob = {}
        for res, prob, success in zip(resilience_vals, coping_probs, coping_success):
            prob_bin = round(prob * 10) / 10  # Bin probabilities in 0.1 increments
            if prob_bin not in success_rate_by_prob:
                success_rate_by_prob[prob_bin] = {'total': 0, 'success': 0}
            success_rate_by_prob[prob_bin]['total'] += 1
            success_rate_by_prob[prob_bin]['success'] += int(success)

        # Calculate success rates by probability bins
        prob_bins = sorted(success_rate_by_prob.keys())
        calibration_data = []
        for prob_bin in prob_bins:
            total = success_rate_by_prob[prob_bin]['total']
            success_count = success_rate_by_prob[prob_bin]['success']
            actual_rate = success_count / total if total > 0 else 0
            calibration_data.append({
                'predicted_prob': prob_bin,
                'actual_success_rate': actual_rate,
                'count': total
            })

        print(f"Resilience vs Coping Probability Correlation: {prob_correlation:.4f}")
        print(f"Sample size: {len(resilience_vals)} events")
        print("\nCoping Probability Calibration:")
        for data in calibration_data:
            print(f"  Predicted {data['predicted_prob']:.1f}: Actual {data['actual_success_rate']:.3f} (n={data['count']})")

        return {
            'resilience_probability_corr': prob_correlation,
            'calibration_data': calibration_data,
            'sample_size': len(resilience_vals)
        }

    def _analyze_correlation_evolution(self) -> Dict[str, Any]:
        """Analyze how correlations evolve over simulation steps."""
        if len(self.correlation_history['step']) == 0:
            return {'error': 'No correlation data available'}

        print("Correlation Evolution Over Time:")
        print("-" * 40)

        # Calculate trends in correlations
        steps = self.correlation_history['step']
        resilience_pss10 = self.correlation_history['resilience_pss10_corr']
        affect_pss10 = self.correlation_history['affect_pss10_corr']
        resilience_coping = self.correlation_history['resilience_coping_corr']
        stress_pss10 = self.correlation_history['stress_pss10_corr']

        # Calculate final correlations
        final_resilience_pss10 = resilience_pss10[-1] if resilience_pss10 else 0
        final_affect_pss10 = affect_pss10[-1] if affect_pss10 else 0
        final_resilience_coping = resilience_coping[-1] if resilience_coping else 0
        final_stress_pss10 = stress_pss10[-1] if stress_pss10 else 0

        print(f"Final Resilience-PSS-10 Correlation: {final_resilience_pss10:.4f}")
        print(f"Final Affect-PSS-10 Correlation: {final_affect_pss10:.4f}")
        print(f"Final Resilience-Coping Correlation: {final_resilience_coping:.4f}")
        print(f"Final Stress-PSS-10 Correlation: {final_stress_pss10:.4f}")

        # Check if correlations are in expected direction (negative for resilience/affect vs PSS-10)
        expected_directions = {
            'resilience_pss10': 'negative',
            'affect_pss10': 'negative',
            'resilience_coping': 'positive',
            'stress_pss10': 'positive'
        }

        print("\nExpected vs Actual Correlation Directions:")
        for corr_name, expected_dir in expected_directions.items():
            final_val = locals()[f'final_{corr_name}']
            actual_dir = 'negative' if final_val < 0 else 'positive' if final_val > 0 else 'none'
            status = '✓' if ((expected_dir == 'negative' and actual_dir == 'negative') or
                           (expected_dir == 'positive' and actual_dir == 'positive')) else '✗'
            print(f"  {corr_name}: Expected {expected_dir}, Got {actual_dir} {status}")

        return {
            'final_correlations': {
                'resilience_pss10': final_resilience_pss10,
                'affect_pss10': final_affect_pss10,
                'resilience_coping': final_resilience_coping,
                'stress_pss10': final_stress_pss10
            },
            'correlation_directions': expected_directions,
            'steps': steps
        }

    def _validate_stress_processing_pipeline(self) -> Dict[str, Any]:
        """Validate that the stress processing pipeline is working correctly."""
        print("Validating Stress Processing Pipeline:")
        print("-" * 40)

        validation_results = {
            'pss10_range_check': True,
            'stress_bounds_check': True,
            'correlation_consistency': True,
            'feedback_loop_check': True
        }

        # 1. PSS-10 Score Range Validation
        pss10_scores = self.debug_data['pss10_score']
        if pss10_scores:
            min_pss10, max_pss10 = min(pss10_scores), max(pss10_scores)
            print(f"PSS-10 Score Range: {min_pss10:.1f} - {max_pss10:.1f}")

            if not (0 <= min_pss10 and max_pss10 <= 40):
                validation_results['pss10_range_check'] = False
                print("  ✗ PSS-10 scores outside valid range [0, 40]")
            else:
                print("  ✓ PSS-10 scores within valid range [0, 40]")
        else:
            validation_results['pss10_range_check'] = False
            print("  ✗ No PSS-10 data available for validation")

        # 2. Stress Level Bounds Validation
        stress_levels = self.debug_data['current_stress']
        controllability_vals = self.debug_data['stress_controllability']
        overload_vals = self.debug_data['stress_overload']

        if stress_levels:
            min_stress, max_stress = min(stress_levels), max(stress_levels)
            print(f"Stress Level Range: {min_stress:.3f} - {max_stress:.3f}")

            if not (0 <= min_stress and max_stress <= 1):
                validation_results['stress_bounds_check'] = False
                print("  ✗ Stress levels outside valid range [0, 1]")
            else:
                print("  ✓ Stress levels within valid range [0, 1]")
        else:
            validation_results['stress_bounds_check'] = False
            print("  ✗ No stress data available for validation")

        # 3. Stress Dimension Validation
        if controllability_vals and overload_vals:
            min_ctrl, max_ctrl = min(controllability_vals), max(controllability_vals)
            min_overload, max_overload = min(overload_vals), max(overload_vals)

            print(f"Stress Controllability Range: {min_ctrl:.3f} - {max_ctrl:.3f}")
            print(f"Stress Overload Range: {min_overload:.3f} - {max_overload:.3f}")

            ctrl_ok = 0 <= min_ctrl and max_ctrl <= 1
            overload_ok = 0 <= min_overload and max_overload <= 1

            if not (ctrl_ok and overload_ok):
                validation_results['stress_bounds_check'] = False
                print("  ✗ Stress dimensions outside valid range [0, 1]")
            else:
                print("  ✓ Stress dimensions within valid range [0, 1]")

        # 4. Theoretical Correlation Validation
        if len(self.correlation_history['step']) > 0:
            final_corr = self.correlation_history['resilience_pss10_corr'][-1]
            print(f"Final Resilience-PSS-10 Correlation: {final_corr:.4f}")

            # Expected negative correlation (higher resilience should correlate with lower PSS-10)
            if final_corr > 0:
                validation_results['correlation_consistency'] = False
                print("  ✗ Expected negative correlation between resilience and PSS-10")
            else:
                print("  ✓ Expected negative correlation between resilience and PSS-10")

        # 5. Feedback Loop Validation
        # Check if PSS-10 feedback is properly updating stress dimensions
        if len(controllability_vals) > 1 and len(pss10_scores) > 1:
            # Simple check: stress dimensions should change when PSS-10 changes
            controllability_changes = [controllability_vals[i] - controllability_vals[i-1]
                                     for i in range(1, len(controllability_vals))]
            pss10_changes = [pss10_scores[i] - pss10_scores[i-1]
                           for i in range(1, len(pss10_scores))]

            if len(controllability_changes) == len(pss10_changes) and len(controllability_changes) > 0:
                feedback_correlation = np.corrcoef(controllability_changes, pss10_changes)[0, 1]
                print(f"PSS-10 Feedback Correlation: {feedback_correlation:.4f}")

                if abs(feedback_correlation) < 0.1:
                    print("  ⚠ PSS-10 feedback loop may not be working effectively")
                else:
                    print("  ✓ PSS-10 feedback loop appears to be functioning")
            else:
                validation_results['feedback_loop_check'] = False
                print("  ✗ Cannot validate feedback loop - insufficient data")

        return validation_results

    def _calculate_and_log_step_metrics(self, step: int, step_description: str = ""):
        """
        Calculate and log comprehensive metrics for the current step.

        Args:
            step: Current simulation step
            step_description: Optional description of the step for logging
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate averages across all agents
        agents = self.model.agents
        if not agents:
            return

        # Basic metrics
        avg_stress = np.mean([agent.current_stress for agent in agents])
        avg_pss10 = np.mean([agent.pss10 for agent in agents])
        avg_affect = np.mean([agent.affect for agent in agents])
        avg_resilience = np.mean([agent.resilience for agent in agents])

        # Event-related metrics
        total_events = sum(len(getattr(agent, 'daily_stress_events', [])) for agent in agents)
        avg_events = total_events / len(agents) if agents else 0

        # Coping success rate (only for agents that had events)
        coping_successes = []
        for agent in agents:
            daily_events = getattr(agent, 'daily_stress_events', [])
            if daily_events:
                # Get the most recent coping success for this agent
                latest_event = daily_events[-1]
                coping_successes.append(latest_event.get('coped_successfully', 0.5))

        avg_coping_success = np.mean(coping_successes) if coping_successes else 0.5

        # Store metrics for analysis
        step_metrics = {
            'timestamp': timestamp,
            'step': step + 1,
            'description': step_description,
            'avg_stress': avg_stress,
            'avg_pss10': avg_pss10,
            'avg_events': avg_events,
            'avg_affect': avg_affect,
            'avg_coping_success': avg_coping_success,
            'avg_resilience': avg_resilience,
            'total_agents': len(agents),
            'total_events': total_events
        }
        self.step_metrics_log.append(step_metrics)

        # Log in the specified format
        step_label = f"Step {step + 1}"
        if step_description:
            step_label += f" ({step_description})"

        print(f"\n[{timestamp}] {step_label}:")
        print(f"- Stress (avg): {avg_stress:.4f}")
        print(f"- PSS-10 (avg): {avg_pss10:.4f}")
        print(f"- Events (avg): {avg_events:.4f}")
        print(f"- Affect (avg): {avg_affect:.4f}")
        print(f"- Coping (avg): {avg_coping_success:.4f}")
        print(f"- Resilience (avg): {avg_resilience:.4f}")

    def export_step_metrics_to_dataframe(self) -> pd.DataFrame:
        """
        Export the step metrics log to a pandas DataFrame for further analysis.

        Returns:
            DataFrame containing all step metrics with timestamps
        """
        if not self.step_metrics_log:
            return pd.DataFrame()

        return pd.DataFrame(self.step_metrics_log)

    def save_step_metrics_to_csv(self, filename: str = None):
        """
        Save the step metrics log to a CSV file.

        Args:
            filename: Optional filename, defaults to timestamped file
        """
        if not self.step_metrics_log:
            print("No step metrics data to save.")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stress_pipeline_metrics_{timestamp}.csv"

        df = self.export_step_metrics_to_dataframe()
        df.to_csv(filename, index=False)
        print(f"Step metrics saved to: {filename}")

    def print_metrics_summary(self):
        """
        Print a summary of the metrics trends across the entire simulation.
        """
        if not self.step_metrics_log:
            print("No step metrics data available for summary.")
            return

        print("\n" + "=" * 80)
        print("STEP METRICS SUMMARY")
        print("=" * 80)

        # Calculate overall averages and trends
        df = self.export_step_metrics_to_dataframe()

        print(f"Simulation Summary ({len(df)} steps recorded):")
        print(f"- Total agents: {df['total_agents'].iloc[0] if len(df) > 0 else 0}")
        print(f"- Average stress level: {df['avg_stress'].mean():.4f}")
        print(f"- Average PSS-10 score: {df['avg_pss10'].mean():.4f}")
        print(f"- Average events per agent: {df['avg_events'].mean():.4f}")
        print(f"- Average affect score: {df['avg_affect'].mean():.4f}")
        print(f"- Average coping success rate: {df['avg_coping_success'].mean():.4f}")
        print(f"- Average resilience score: {df['avg_resilience'].mean():.4f}")

        # Show trends (first vs last 10 steps)
        if len(df) >= 10:
            first_10 = df.head(10)
            last_10 = df.tail(10)

            print("\nTrends (comparing first 10 vs last 10 steps):")
            print(f"- Stress: {first_10['avg_stress'].mean():.4f} → {last_10['avg_stress'].mean():.4f}")
            print(f"- PSS-10: {first_10['avg_pss10'].mean():.4f} → {last_10['avg_pss10'].mean():.4f}")
            print(f"- Events: {first_10['avg_events'].mean():.4f} → {last_10['avg_events'].mean():.4f}")
            print(f"- Affect: {first_10['avg_affect'].mean():.4f} → {last_10['avg_affect'].mean():.4f}")
            print(f"- Coping: {first_10['avg_coping_success'].mean():.4f} → {last_10['avg_coping_success'].mean():.4f}")
            print(f"- Resilience: {first_10['avg_resilience'].mean():.4f} → {last_10['avg_resilience'].mean():.4f}")


def run_stress_pipeline_debug_demo():
    """
    Run the complete stress processing pipeline debug demonstration.

    Returns:
        Dictionary containing all debugging results
    """
    # Create debugger with extended simulation for correlation development
    debugger = StressPipelineDebugger(
        num_agents=20,  # Sufficient for correlation analysis
        max_steps=50,   # Extended simulation for correlations to develop
        seed=42         # Reproducible results
    )

    # Run the debugging simulation
    results = debugger.run_debugging_simulation()

    # Print metrics summary
    debugger.print_metrics_summary()

    # Print final summary
    print("\n" + "=" * 80)
    print("DEBUG DEMO SUMMARY")
    print("=" * 80)

    analysis = results['analysis_results']

    # Summary of key findings
    print("\nKEY FINDINGS:")
    print("-" * 50)

    # 1. Resilience-Coping Relationship
    resilience_coping = analysis['resilience_coping']
    if 'error' not in resilience_coping:
        corr = resilience_coping['correlation']
        print(f"1. Resilience-Coping Correlation: {corr:.4f} {'✓' if corr > 0 else '✗'}")

    # 2. PSS-10-Stress Reflection
    pss10_stress = analysis['pss10_stress']
    if 'error' not in pss10_stress:
        corr = pss10_stress['pss10_stress_corr']
        print(f"2. PSS-10-Stress Correlation: {corr:.4f} {'✓' if corr > 0 else '✗'}")

    # 3. Coping Probability Mechanism
    coping_prob = analysis['coping_probability']
    if 'error' not in coping_prob:
        corr = coping_prob['resilience_probability_corr']
        print(f"3. Resilience-Coping Probability Correlation: {corr:.4f} {'✓' if corr > 0 else '✗'}")

    # 4. Final Correlation Status
    corr_evolution = analysis['correlation_evolution']
    if 'error' not in corr_evolution:
        final_corr = corr_evolution['final_correlations']['resilience_pss10']
        print(f"4. Final Resilience-PSS-10 Correlation: {final_corr:.4f} {'✓' if final_corr < 0 else '✗'}")

    # 5. Pipeline Validation
    pipeline_validation = analysis['pipeline_validation']
    valid_count = sum(pipeline_validation.values())
    total_checks = len(pipeline_validation)
    print(f"5. Pipeline Validation: {valid_count}/{total_checks} checks passed")

    print("\n" + "=" * 80)
    print("DEBUG DEMO COMPLETED")
    print("=" * 80)
    print("Use the returned results dictionary for further analysis")
    print("Key data available:")
    print("  - results['debug_data']: Step-by-step debugging data")
    print("  - results['correlation_history']: Correlation evolution over time")
    print("  - results['analysis_results']: Comprehensive analysis results")
    print("  - results['step_metrics_log']: Timestamped step-by-step metrics with averages")

    return results


if __name__ == "__main__":
    # Run the complete debugging demo
    results = run_stress_pipeline_debug_demo()
