"""
Configuration module for Agent-Based Mental Health Simulation.

This module provides centralized configuration management using environment variables
with python-dotenv for loading and type conversion with proper error handling.
"""

import os
import logging
from typing import Optional, Union
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an error in configuration loading or validation."""
    pass


class Config:
    """
    Centralized configuration class for the ABM simulation.

    Loads environment variables with type conversion and validation.
    Provides fallback defaults based on current hardcoded values.
    """

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration by loading environment variables.

        Args:
            env_file: Path to .env file (default: .env in project root)
        """
        self._load_environment(env_file or ".env")
        self._config = {}
        self._load_all_parameters()

    def _load_environment(self, env_file: str) -> None:
        """Load environment variables from .env file if it exists."""
        env_path = Path(env_file)

        if env_path.exists():
            try:
                # Load environment variables from file
                load_dotenv(env_path)
                logger.info(f"Loaded environment from {env_path.absolute()}")
            except ImportError:
                logger.warning(f"python-dotenv not installed, using system environment variables")
        else:
            logger.warning(f"Environment file {env_path.absolute()} not found, using system environment variables")

    def _get_env_value(self, key: str, expected_type: type, default_value: Union[str, int, float, bool], required: bool = False) -> Union[int, float, bool, str]:
        """
        Get environment variable with type conversion and validation.

        Args:
            key: Environment variable name
            expected_type: Expected type (int, float, bool, str)
            default_value: Default value if not found
            required: Whether this variable is required

        Returns:
            Converted value in the expected type

        Raises:
            ConfigurationError: If required variable is missing or conversion fails
        """
        value = os.getenv(key)

        if value is None:
            if required:
                raise ConfigurationError(f"Required environment variable '{key}' is not set")
            logger.debug(f"Using default value for '{key}': {default_value}")
            return default_value

        try:
            if expected_type == int:
                return int(float(value))  # Handle cases like "42.0"
            elif expected_type == float:
                return float(value)
            elif expected_type == bool:
                if value.lower() not in ('true', '1', 'yes', 'on', 'false', '0', 'no', 'off'):
                    raise ValueError(f"Invalid boolean value: '{value}'")
                return value.lower() in ('true', '1', 'yes', 'on')
            else:
                return value
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Invalid value for '{key}': '{value}'. Expected {expected_type.__name__}. {e}")

    def _get_env_array(self, key: str, expected_type: type, default_array: list, required: bool = False, expected_length: int = None) -> list:
        """
        Get environment variable as array with type conversion and validation.

        Supports both bracket notation: [2.1, 1.8, 2.3, ...]
        and space-separated format: 2.1 1.8 2.3 ... (for backward compatibility)

        Args:
            key: Environment variable name
            expected_type: Expected type for array elements (int, float)
            default_array: Default array if not found
            required: Whether this variable is required
            expected_length: Expected length of array (optional validation)

        Returns:
            List of converted values in the expected type

        Raises:
            ConfigurationError: If required variable is missing, conversion fails, or length validation fails
        """
        value = os.getenv(key)

        if value is None:
            if required:
                raise ConfigurationError(f"Required environment variable '{key}' is not set")
            logger.debug(f"Using default array for '{key}': {default_array}")
            return default_array

        try:
            # Determine format and parse accordingly
            if value is None:
                stripped_value = ""
            else:
                stripped_value = value.strip()

            if stripped_value.startswith('[') and stripped_value.endswith(']'):
                # Bracket notation: [2.1, 1.8, 2.3, ...]
                # Remove brackets and split by comma
                inner_value = stripped_value[1:-1].strip()
                if inner_value:
                    elements = [elem.strip() for elem in inner_value.split(',')]
                else:
                    elements = []
            else:
                # Space-separated format (backward compatibility): 2.1 1.8 2.3 ...
                elements = stripped_value.split()

            result = []

            for element in elements:
                # Skip empty elements (handles extra whitespace or commas)
                element = element.strip()
                if not element:
                    continue

                if expected_type == int:
                    result.append(int(float(element)))  # Handle cases like "2.0"
                elif expected_type == float:
                    result.append(float(element))
                else:
                    raise ConfigurationError(f"Unsupported array type: {expected_type}")

            # Validate length if specified
            if expected_length is not None and len(result) != expected_length:
                raise ConfigurationError(f"Array '{key}' must have exactly {expected_length} elements, got {len(result)}")

            return result

        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Invalid array value for '{key}': '{value}'. Expected array of {expected_type.__name__}s. {e}")

    def _load_all_parameters(self) -> None:
        """Load all configuration parameters from environment variables."""

        # ==============================================
        # SIMULATION AND NETWORK PARAMETERS
        # ==============================================
        self.num_agents = self._get_env_value('SIMULATION_NUM_AGENTS', int, 20)
        self.max_days = self._get_env_value('SIMULATION_MAX_DAYS', int, 100)
        self.seed = self._get_env_value('SIMULATION_SEED', int, 42)

        # Network topology parameters (Watts-Strogatz)
        self.network_watts_k = self._get_env_value('NETWORK_WATTS_K', int, 4)
        self.network_watts_p = self._get_env_value('NETWORK_WATTS_P', float, 0.1)

        # New network adaptation parameters
        self.network_adaptation_threshold = self._get_env_value('NETWORK_ADAPTATION_THRESHOLD', int, 3)
        self.network_rewire_probability = self._get_env_value('NETWORK_REWIRE_PROBABILITY', float, 0.01)
        self.network_homophily_strength = self._get_env_value('NETWORK_HOMOPHILY_STRENGTH', float, 0.7)

        # ==============================================
        # AGENT STATE AND BEHAVIOR PARAMETERS
        # ==============================================
        self.agent_initial_resilience = self._get_env_value('AGENT_INITIAL_RESILIENCE', float, 0.5)
        self.agent_initial_affect = self._get_env_value('AGENT_INITIAL_AFFECT', float, 0.0)
        self.agent_initial_resources = self._get_env_value('AGENT_INITIAL_RESOURCES', float, 0.6)

        self.agent_stress_probability = self._get_env_value('AGENT_STRESS_PROBABILITY', float, 0.5)
        self.agent_coping_success_rate = self._get_env_value('AGENT_COPING_SUCCESS_RATE', float, 0.5)
        self.agent_subevents_per_day = self._get_env_value('AGENT_SUBEVENTS_PER_DAY', int, 3)
        self.agent_resource_cost = self._get_env_value('AGENT_RESOURCE_COST', float, 0.1)

        # ==============================================
        # STRESS EVENT PARAMETERS
        # ==============================================
        self.stress_controllability_mean = self._get_env_value('STRESS_CONTROLLABILITY_MEAN', float, 0.5)
        self.stress_overload_mean = self._get_env_value('STRESS_OVERLOAD_MEAN', float, 0.5)
        self.stress_beta_alpha = self._get_env_value('STRESS_BETA_ALPHA', float, 2.0)
        self.stress_beta_beta = self._get_env_value('STRESS_BETA_BETA', float, 2.0)

        # ==============================================
        # APPRAISAL AND THRESHOLD PARAMETERS
        # ==============================================
        self.appraisal_omega_c = self._get_env_value('APPRAISAL_OMEGA_C', float, 1.0)
        self.appraisal_omega_o = self._get_env_value('APPRAISAL_OMEGA_O', float, 1.0)
        self.appraisal_bias = self._get_env_value('APPRAISAL_BIAS', float, 0.0)
        self.appraisal_gamma = self._get_env_value('APPRAISAL_GAMMA', float, 6.0)

        self.threshold_base_threshold = self._get_env_value('THRESHOLD_BASE_THRESHOLD', float, 0.5)
        self.threshold_challenge_scale = self._get_env_value('THRESHOLD_CHALLENGE_SCALE', float, 0.15)
        self.threshold_hindrance_scale = self._get_env_value('THRESHOLD_HINDRANCE_SCALE', float, 0.25)
        self.threshold_stress_threshold = self._get_env_value('THRESHOLD_STRESS_THRESHOLD', float, 0.3)
        self.threshold_affect_threshold = self._get_env_value('THRESHOLD_AFFECT_THRESHOLD', float, 0.3)

        self.stress_alpha_challenge = self._get_env_value('STRESS_ALPHA_CHALLENGE', float, 0.8)
        self.stress_alpha_hindrance = self._get_env_value('STRESS_ALPHA_HINDRANCE', float, 1.2)
        self.stress_delta = self._get_env_value('STRESS_DELTA', float, 0.2)

        # ==============================================
        # PSS-10 SCORE GENERATION PARAMETERS
        # ==============================================
        self.pss10_item_means = self._get_env_array('PSS10_ITEM_MEAN', float, [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5], expected_length=10)
        self.pss10_item_sds = self._get_env_array('PSS10_ITEM_SD', float, [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8], expected_length=10)

        # PSS-10 bifactor model parameters
        self.pss10_load_controllability = self._get_env_array('PSS10_LOAD_CONTROLLABILITY', float, [0.2, 0.8, 0.1, 0.7, 0.6, 0.1, 0.8, 0.6, 0.7, 0.1], expected_length=10)
        self.pss10_load_overload = self._get_env_array('PSS10_LOAD_OVERLOAD', float, [0.7, 0.3, 0.8, 0.2, 0.4, 0.9, 0.2, 0.3, 0.4, 0.9], expected_length=10)
        self.pss10_bifactor_correlation = self._get_env_value('PSS10_BIFACTOR_COR', float, 0.3)

        # PSS-10 standard deviations for controllability and overload factors
        self.pss10_controllability_sd = self._get_env_value('PSS10_CONTROLLABILITY_SD', float, 1.0)
        self.pss10_overload_sd = self._get_env_value('PSS10_OVERLOAD_SD', float, 1.0)

        # PSS-10 threshold for stress classification
        self.pss10_threshold = self._get_env_value('PSS10_THRESHOLD', int, 27)

        # New coping probability mechanism parameters
        self.coping_base_probability = self._get_env_value('COPING_BASE_PROBABILITY', float, 0.5)
        self.coping_social_influence = self._get_env_value('COPING_SOCIAL_INFLUENCE', float, 0.1)
        self.coping_challenge_bonus = self._get_env_value('COPING_CHALLENGE_BONUS', float, 0.2)
        self.coping_hindrance_penalty = self._get_env_value('COPING_HINDRANCE_PENALTY', float, 0.3)

        # ==============================================
        # SOCIAL INTERACTION PARAMETERS
        # ==============================================
        self.interaction_influence_rate = self._get_env_value('INTERACTION_INFLUENCE_RATE', float, 0.05)
        self.interaction_resilience_influence = self._get_env_value('INTERACTION_RESILIENCE_INFLUENCE', float, 0.05)
        self.interaction_max_neighbors = self._get_env_value('INTERACTION_MAX_NEIGHBORS', int, 10)

        # ==============================================
        # AFFECT AND RESILIENCE DYNAMICS PARAMETERS
        # ==============================================
        self.affect_peer_influence_rate = self._get_env_value('AFFECT_PEER_INFLUENCE_RATE', float, 0.1)
        self.affect_event_appraisal_rate = self._get_env_value('AFFECT_EVENT_APPRAISAL_RATE', float, 0.15)
        self.affect_homeostatic_rate = self._get_env_value('AFFECT_HOMEOSTATIC_RATE', float, 0.5)
        self.resilience_homeostatic_rate = self._get_env_value('RESILIENCE_HOMEOSTATIC_RATE', float, 0.05)

        self.resilience_coping_success_rate = self._get_env_value('RESILIENCE_COPING_SUCCESS_RATE', float, 0.1)
        self.resilience_social_support_rate = self._get_env_value('RESILIENCE_SOCIAL_SUPPORT_RATE', float, 0.08)
        self.resilience_overload_threshold = self._get_env_value('RESILIENCE_OVERLOAD_THRESHOLD', int, 3)

        # New resilience boost rate
        self.resilience_boost_rate = self._get_env_value('RESILIENCE_BOOST_RATE', float, 0.1)

        self.influencing_neighbors = self._get_env_value('N_INFLUENCING_NEIGHBORS', int, 5)
        self.influencing_hindrance = self._get_env_value('N_INFLUENCING_HINDRANCE', int, 3)

        # Stress and affect dynamics parameters
        self.stress_decay_rate = self._get_env_value('STRESS_DECAY_RATE', float, 0.05)

        # ==============================================
        # RESOURCE DYNAMICS PARAMETERS
        # ==============================================
        self.protective_social_support = self._get_env_value('PROTECTIVE_SOCIAL_SUPPORT', float, 0.5)
        self.protective_family_support = self._get_env_value('PROTECTIVE_FAMILY_SUPPORT', float, 0.5)
        self.protective_formal_intervention = self._get_env_value('PROTECTIVE_FORMAL_INTERVENTION', float, 0.5)
        self.protective_psychological_capital = self._get_env_value('PROTECTIVE_PSYCHOLOGICAL_CAPITAL', float, 0.5)

        self.resource_base_regeneration = self._get_env_value('RESOURCE_BASE_REGENERATION', float, 0.05)
        self.resource_allocation_cost = self._get_env_value('RESOURCE_ALLOCATION_COST', float, 0.15)
        self.resource_cost_exponent = self._get_env_value('RESOURCE_COST_EXPONENT', float, 1.5)

        # New protective factor improvement rate
        self.protective_improvement_rate = self._get_env_value('PROTECTIVE_IMPROVEMENT_RATE', float, 0.5)

        # ==============================================
        # MATHEMATICAL UTILITY PARAMETERS
        # ==============================================
        self.utility_softmax_temperature = self._get_env_value('UTILITY_SOFTMAX_TEMPERATURE', float, 1.0)

        # ==============================================
        # OUTPUT AND LOGGING CONFIGURATION
        # ==============================================
        self.log_level = self._get_env_value('LOG_LEVEL', str, 'INFO')

        self.output_results_dir = self._get_env_value('OUTPUT_RESULTS_DIR', str, 'data/processed')
        self.output_raw_dir = self._get_env_value('OUTPUT_RAW_DIR', str, 'data/raw')
        self.output_logs_dir = self._get_env_value('OUTPUT_LOGS_DIR', str, 'logs')

        self.output_save_time_series = self._get_env_value('OUTPUT_SAVE_TIME_SERIES', bool, True)
        self.output_save_network_snapshots = self._get_env_value('OUTPUT_SAVE_NETWORK_SNAPSHOTS', bool, True)
        self.output_save_summary_statistics = self._get_env_value('OUTPUT_SAVE_SUMMARY_STATISTICS', bool, True)

        # Store all config in a dictionary for easy access
        self._config = {
            'simulation': {
                'num_agents': self.num_agents,
                'max_days': self.max_days,
                'seed': self.seed,
            },
            'network': {
                'watts_k': self.network_watts_k,
                'watts_p': self.network_watts_p,
                'adaptation_threshold': self.network_adaptation_threshold,
                'rewire_probability': self.network_rewire_probability,
                'homophily_strength': self.network_homophily_strength,
            },
            'agent': {
                'initial_resilience': self.agent_initial_resilience,
                'initial_affect': self.agent_initial_affect,
                'initial_resources': self.agent_initial_resources,
                'stress_probability': self.agent_stress_probability,
                'coping_success_rate': self.agent_coping_success_rate,
                'subevents_per_day': self.agent_subevents_per_day,
                'resource_cost': self.agent_resource_cost,
            },
            'coping': {
                'base_probability': self.coping_base_probability,
                'social_influence': self.coping_social_influence,
                'challenge_bonus': self.coping_challenge_bonus,
                'hindrance_penalty': self.coping_hindrance_penalty,
            },
            'stress': {
                'controllability_mean': self.stress_controllability_mean,
                'overload_mean': self.stress_overload_mean,
                'beta_alpha': self.stress_beta_alpha,
                'beta_beta': self.stress_beta_beta,
            },
            'appraisal': {
                'omega_c': self.appraisal_omega_c,
                'omega_o': self.appraisal_omega_o,
                'bias': self.appraisal_bias,
                'gamma': self.appraisal_gamma,
            },
            'threshold': {
                'base_threshold': self.threshold_base_threshold,
                'challenge_scale': self.threshold_challenge_scale,
                'hindrance_scale': self.threshold_hindrance_scale,
                'stress_threshold': self.threshold_stress_threshold,
                'affect_threshold': self.threshold_affect_threshold,
            },
            'stress_params': {
                'alpha_challenge': self.stress_alpha_challenge,
                'alpha_hindrance': self.stress_alpha_hindrance,
                'delta': self.stress_delta,
            },
            'pss10': {
                'item_means': self.pss10_item_means,
                'item_sds': self.pss10_item_sds,
                'load_controllability': self.pss10_load_controllability,
                'load_overload': self.pss10_load_overload,
                'bifactor_correlation': self.pss10_bifactor_correlation,
                'controllability_sd': self.pss10_controllability_sd,
                'overload_sd': self.pss10_overload_sd,
                'threshold': self.pss10_threshold,
            },
            'interaction': {
                'influence_rate': self.interaction_influence_rate,
                'resilience_influence': self.interaction_resilience_influence,
                'max_neighbors': self.interaction_max_neighbors,
            },
            'affect_dynamics': {
                'peer_influence_rate': self.affect_peer_influence_rate,
                'event_appraisal_rate': self.affect_event_appraisal_rate,
                'homeostatic_rate': self.affect_homeostatic_rate,
            },
            'resilience_dynamics': {
                'coping_success_rate': self.resilience_coping_success_rate,
                'social_support_rate': self.resilience_social_support_rate,
                'overload_threshold': self.resilience_overload_threshold,
                'homeostatic_rate': self.resilience_homeostatic_rate,
                'boost_rate': self.resilience_boost_rate,
            },
            'influence': {
                'influencing_neighbors': self.influencing_neighbors,
                'influencing_hindrance': self.influencing_hindrance,
            },
            'dynamics': {
                'stress_decay_rate': self.stress_decay_rate,
            },
            'protective': {
                'social_support': self.protective_social_support,
                'family_support': self.protective_family_support,
                'formal_intervention': self.protective_formal_intervention,
                'psychological_capital': self.protective_psychological_capital,
            },
            'resource': {
                'base_regeneration': self.resource_base_regeneration,
                'allocation_cost': self.resource_allocation_cost,
                'cost_exponent': self.resource_cost_exponent,
                'protective_improvement_rate': self.protective_improvement_rate,
            },
            'utility': {
                'softmax_temperature': self.utility_softmax_temperature,
            },
            'output': {
                'log_level': self.log_level,
                'results_dir': self.output_results_dir,
                'raw_dir': self.output_raw_dir,
                'logs_dir': self.output_logs_dir,
                'save_time_series': self.output_save_time_series,
                'save_network_snapshots': self.output_save_network_snapshots,
                'save_summary_statistics': self.output_save_summary_statistics,
            }
        }

    def get(self, section: str, key: str = None):
        """
        Get configuration value(s).

        Args:
            section: Configuration section name
            key: Specific key within section (if None, returns entire section)

        Returns:
            Configuration value or section dictionary

        Raises:
            ConfigurationError: If section or key doesn't exist
        """
        if section not in self._config:
            raise ConfigurationError(f"Configuration section '{section}' not found")

        if key is None:
            return self._config[section]

        if key not in self._config[section]:
            raise ConfigurationError(f"Configuration key '{key}' not found in section '{section}'")

        return self._config[section][key]

    def validate(self) -> None:
        """
        Validate configuration parameters for consistency and reasonable ranges.

        Raises:
            ConfigurationError: If validation fails
        """
        # Network validation
        if self.network_watts_k < 2:
            raise ConfigurationError("Network k parameter must be >= 2")
        if not (0 <= self.network_watts_p <= 1):
            raise ConfigurationError("Network p parameter must be in [0, 1]")
        if self.network_adaptation_threshold < 1:
            raise ConfigurationError("Network adaptation threshold must be >= 1")
        if not (0 <= self.network_rewire_probability <= 1):
            raise ConfigurationError("Network rewiring probability must be in [0, 1]")
        if not (0 <= self.network_homophily_strength <= 1):
            raise ConfigurationError("Network homophily strength must be in [0, 1]")

        # Agent validation
        if not (0 <= self.agent_initial_resilience <= 1):
            raise ConfigurationError("Agent initial resilience must be in [0, 1]")
        if not (-1 <= self.agent_initial_affect <= 1):
            raise ConfigurationError("Agent initial affect must be in [-1, 1]")
        if not (0 <= self.agent_initial_resources <= 1):
            raise ConfigurationError("Agent initial resources must be in [0, 1]")
        if not (0 <= self.agent_stress_probability <= 1):
            raise ConfigurationError("Agent stress probability must be in [0, 1]")

        # Stress validation
        for param in [self.stress_controllability_mean, self.stress_overload_mean]:
            if not (0 <= param <= 1):
                raise ConfigurationError("Stress event means must be in [0, 1]")

        # PSS-10 validation
        if len(self.pss10_item_means) != 10:
            raise ConfigurationError("PSS-10 item means must have exactly 10 values")

        if len(self.pss10_item_sds) != 10:
            raise ConfigurationError("PSS-10 item standard deviations must have exactly 10 values")

        if len(self.pss10_load_controllability) != 10:
            raise ConfigurationError("PSS-10 controllability loadings must have exactly 10 values")

        if len(self.pss10_load_overload) != 10:
            raise ConfigurationError("PSS-10 overload loadings must have exactly 10 values")

        for i, mean_val in enumerate(self.pss10_item_means):
            if not (0 <= mean_val <= 4):
                raise ConfigurationError(f"PSS-10 item mean at index {i} must be in [0, 4], got {mean_val}")

        for i, sd_val in enumerate(self.pss10_item_sds):
            if sd_val <= 0:
                raise ConfigurationError(f"PSS-10 item standard deviation at index {i} must be positive, got {sd_val}")

        for i, loading_val in enumerate(self.pss10_load_controllability):
            if not (0 <= loading_val <= 1):
                raise ConfigurationError(f"PSS-10 controllability loading at index {i} must be in [0, 1], got {loading_val}")

        for i, loading_val in enumerate(self.pss10_load_overload):
            if not (0 <= loading_val <= 1):
                raise ConfigurationError(f"PSS-10 overload loading at index {i} must be in [0, 1], got {loading_val}")

        if not (-1 <= self.pss10_bifactor_correlation <= 1):
            raise ConfigurationError(f"PSS-10 bifactor correlation must be in [-1, 1], got {self.pss10_bifactor_correlation}")

        # PSS-10 standard deviation validation
        if self.pss10_controllability_sd <= 0:
            raise ConfigurationError(f"PSS-10 controllability SD must be positive, got {self.pss10_controllability_sd}")
        if self.pss10_overload_sd <= 0:
            raise ConfigurationError(f"PSS-10 overload SD must be positive, got {self.pss10_overload_sd}")

        # PSS-10 bifactor model validation
        if len(self.pss10_load_controllability) != 10:
            raise ConfigurationError("PSS-10 controllability loadings must have exactly 10 values")

        if len(self.pss10_load_overload) != 10:
            raise ConfigurationError("PSS-10 overload loadings must have exactly 10 values")

        for i, load_val in enumerate(self.pss10_load_controllability):
            if not (0 <= load_val <= 1):
                raise ConfigurationError(f"PSS-10 controllability loading at index {i} must be in [0, 1], got {load_val}")

        for i, load_val in enumerate(self.pss10_load_overload):
            if not (0 <= load_val <= 1):
                raise ConfigurationError(f"PSS-10 overload loading at index {i} must be in [0, 1], got {load_val}")

        if not (-1 <= self.pss10_bifactor_correlation <= 1):
            raise ConfigurationError(f"PSS-10 bifactor correlation must be in [-1, 1], got {self.pss10_bifactor_correlation}")

        # Threshold validation
        if not (0 <= self.threshold_base_threshold <= 1):
            raise ConfigurationError("Base threshold must be in [0, 1]")
        if not (0 <= self.threshold_stress_threshold <= 1):
            raise ConfigurationError("Stress threshold must be in [0, 1]")
        if not (0 <= self.threshold_affect_threshold <= 1):
            raise ConfigurationError("Affect threshold must be in [0, 1]")

        # Coping validation
        if not (0 <= self.coping_base_probability <= 1):
            raise ConfigurationError("Coping base probability must be in [0, 1]")
        if not (0 <= self.coping_social_influence <= 1):
            raise ConfigurationError("Coping social influence must be in [0, 1]")
        if not (0 <= self.coping_challenge_bonus <= 1):
            raise ConfigurationError("Coping challenge bonus must be in [0, 1]")
        if not (0 <= self.coping_hindrance_penalty <= 1):
            raise ConfigurationError("Coping hindrance penalty must be in [0, 1]")

        # Dynamics validation
        if not (0 <= self.stress_decay_rate <= 1):
            raise ConfigurationError("Stress decay rate must be in [0, 1]")

        # Protective factors validation
        for param in [self.protective_social_support, self.protective_family_support,
                     self.protective_formal_intervention, self.protective_psychological_capital]:
            if not (0 <= param <= 1):
                raise ConfigurationError("Protective factors must be in [0, 1]")

        # Resource validation
        if self.resource_base_regeneration < 0:
            raise ConfigurationError("Resource regeneration rate must be >= 0")
        if self.resource_allocation_cost < 0:
            raise ConfigurationError("Resource allocation cost must be >= 0")
        if self.resource_cost_exponent < 1:
            raise ConfigurationError("Resource cost exponent must be >= 1")
        if not (0 <= self.protective_improvement_rate <= 1):
            raise ConfigurationError("Protective improvement rate must be in [0, 1]")

        # Utility validation
        if self.utility_softmax_temperature <= 0:
            raise ConfigurationError("Softmax temperature must be > 0")

        logger.info("Configuration validation passed")

    def print_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("AGENT-BASED MENTAL HEALTH SIMULATION - CONFIGURATION")
        print("="*60)

        for section_name, section_data in self._config.items():
            print(f"\n{section_name.upper()}:")
            print("-" * len(section_name))
            for key, value in section_data.items():
                print(f"  {key}: {value}")

        print("\n" + "="*60 + "\n")


# Global configuration instance
config = None


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        env_file: Path to .env file (default: .env in project root)

    Returns:
        Config instance
    """
    global config
    if config is None:
        config = Config(env_file)
        config.validate()
    return config


def reload_config(env_file: Optional[str] = None) -> Config:
    """
    Reload configuration from environment.

    Args:
        env_file: Path to .env file (default: .env in project root)

    Returns:
        New Config instance
    """
    global config
    config = Config(env_file)
    config.validate()
    return config


if __name__ == "__main__":
    # Example usage and testing
    try:
        cfg = get_config()
        cfg.print_summary()
        print("Configuration loaded successfully!")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        exit(1)
