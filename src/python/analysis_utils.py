import pandas as pd
import numpy as np
import logging
import os


def compute_stats(series: pd.Series) -> tuple:
    """Compute mean, std, min, max for a series, handling NaN."""
    if series.empty or series.isna().all():
        return np.nan, np.nan, np.nan, np.nan
    mean_val = series.mean()
    std_val = series.std()
    min_val = series.min()
    max_val = series.max()
    return mean_val, std_val, min_val, max_val


def format_mean_sd(mean_val, std_val) -> str:
    """Format mean ± std."""
    if pd.isna(mean_val) or pd.isna(std_val):
        return "N/A"
    return f"{mean_val:.3f} ± {std_val:.3f}"


def format_min_max(min_val, max_val) -> str:
    """Format min - max."""
    if pd.isna(min_val) or pd.isna(max_val):
        return "N/A"
    return f"{min_val:.2f} - {max_val:.2f}"


def assess_stability(cv: float, relative_range: float) -> str:
    """Assess stability level based on CV and relative range."""
    if cv < 0.1 and relative_range < 0.5:
        return "high"
    elif cv < 0.3 and relative_range < 1.0:
        return "moderate"
    else:
        return "low"


def get_stability_interpretation(stability: str, metric_type: str) -> str:
    """Get concise interpretation based on stability and metric type."""
    interpretations = {
        "PSS-10": {"high": "Stable stress perception", "moderate": "Moderate stress variation", "low": "Variable stress levels"},
        "resilience": {"high": "Uniform coping capacity", "moderate": "Balanced resilience differences", "low": "Diverse adaptive capacities"},
        "affect": {"high": "Stable emotional state", "moderate": "Controlled mood variation", "low": "Significant mood fluctuations"},
        "coping_success": {"high": "Reliable coping outcomes", "moderate": "Balanced success variation", "low": "Inconsistent coping rates"},
        "resources": {"high": "Stable resource levels", "moderate": "Controlled resource dynamics", "low": "Fluctuating resource capacity"},
        "current_stress": {"high": "Steady stress conditions", "moderate": "Controlled intensity variation", "low": "Fluctuating stress pressure"},
        "challenge_hindrance": {"high": "Balanced appraisal pattern", "moderate": "Moderate appraisal balance", "low": "Unbalanced cognitive processing"},
        "hindrance_streak": {"high": "Consistent stress sequences", "moderate": "Occasional extended periods", "low": "Unpredictable stress patterns"},
        "coping_events": {"high": "Uniform behavioral patterns", "moderate": "Reasonable individual variation", "low": "Diverse behavioral responses"},
        "social_exchanges": {"high": "Consistent network engagement", "moderate": "Balanced interaction variation", "low": "Diverse network participation"},
        "support_transactions": {"high": "Consistent mutual aid", "moderate": "Controlled assistance variation", "low": "Inconsistent help-seeking"},
        "controllability": {"high": "Consistent agency beliefs", "moderate": "Balanced control perceptions", "low": "Diverse agency experiences"},
        "overload": {"high": "Consistent capacity assessments", "moderate": "Controlled burden variation", "low": "Diverse capacity challenges"}
    }
    return interpretations.get(metric_type, {}).get(stability, "Variable patterns observed")


def generate_interpretation(description: str, mean_val: float, std_val: float, min_val: float, max_val: float) -> str:
    """Generate concise interpretation based on stats, focusing on stability."""
    if pd.isna(mean_val) or pd.isna(std_val) or pd.isna(min_val) or pd.isna(max_val):
        return "Data unavailable"

    # Calculate stability metrics
    cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
    range_val = max_val - min_val
    relative_range = range_val / abs(mean_val) if mean_val != 0 else float('inf')

    stability = assess_stability(cv, relative_range)

    # Determine metric type
    if "PSS-10" in description:
        metric_type = "PSS-10"
    elif "resilience" in description:
        metric_type = "resilience"
    elif "affect" in description:
        metric_type = "affect"
    elif "coping_success" in description or "successful coping events" in description:
        metric_type = "coping_success" if "Proportion" in description else "coping_events"
    elif "resources" in description:
        metric_type = "resources"
    elif "current_stress" in description:
        metric_type = "current_stress"
    elif "challenge vs. hindrance" in description or "challenge–hindrance difference" in description:
        metric_type = "challenge_hindrance"
    elif "hindrance_streak" in description or "consecutive_hindrances" in description:
        metric_type = "hindrance_streak"
    elif "social exchanges" in description:
        metric_type = "social_exchanges"
    elif "support transactions" in description:
        metric_type = "support_transactions"
    elif "controllability" in description or "stress_controllability" in description:
        metric_type = "controllability"
    elif "overload" in description or "stress_overload" in description:
        metric_type = "overload"
    else:
        metric_type = "default"

    return get_stability_interpretation(stability, metric_type)


def analyze_simulation_data(model_df: pd.DataFrame, agent_df: pd.DataFrame, output_dir: str, prefix: str):
    """
    Perform statistical analysis on model and agent data, computing specific metrics and saving as CSV tables.

    Generates two CSV files: one for model-level aggregated data and one for agent-level per-step data,
    with columns: Description, Mean ± SD, Min - Max, Interpretation.

    Parameters:
    - model_df (pd.DataFrame): DataFrame containing model-level data (not directly used for metrics, but passed for consistency)
    - agent_df (pd.DataFrame): DataFrame containing agent-level data
    - output_dir (str): Directory path where CSV files will be saved
    - prefix (str): Prefix for output CSV filenames

    Returns:
    None
    """
    logger = logging.getLogger(__name__)

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define model data metrics (aggregated across agents)
        model_metrics = [
            {
                'description': 'Mean perceived stress (PSS-10) across agents',
                'column': 'avg_pss10',
                'agg_func': None,  # Direct on model_df
            },
            {
                'description': 'Average resilience level',
                'column': 'avg_resilience',
                'agg_func': None,
            },
            {
                'description': 'Mean affect (emotional valence)',
                'column': 'avg_affect',
                'agg_func': None,
            },
            {
                'description': 'Proportion of successful coping events',
                'column': 'coping_success_rate',
                'agg_func': None,
            },
            {
                'description': 'Mean psychological resources',
                'column': 'avg_resources',
                'agg_func': None,
            },
            {
                'description': 'Average current stress level',
                'column': 'avg_stress',
                'agg_func': None,
            },
            {
                'description': 'Mean challenge vs. hindrance appraisal',
                'column': ['avg_challenge', 'avg_hindrance'],
                'agg_func': None,
            },
            {
                'description': 'Challenge–hindrance difference',
                'column': 'challenge_hindrance_ratio',
                'agg_func': None,
            },
            {
                'description': 'Mean sequence of hindrance events',
                'column': 'avg_consecutive_hindrances',
                'agg_func': None,
            },
            {
                'description': 'Count of successful coping events per agent',
                'column': 'successful_coping',
                'agg_func': 'sum_per_agent',
            },
            {
                'description': 'Number of social exchanges per agent',
                'column': 'social_interactions',
                'agg_func': 'sum_per_agent',
            },
            {
                'description': 'Number of support transactions',
                'column': 'support_exchanges',
                'agg_func': 'sum_per_agent',
            }
        ]

        # Define agent data metrics (per agent-step)
        agent_metrics = [
            {
                'description': 'Perceived stress (PSS-10) per agent-step',
                'column': 'pss10',
            },
            {
                'description': 'Instantaneous resilience level',
                'column': 'resilience',
            },
            {
                'description': 'Current emotional state',
                'column': 'affect',
            },
            {
                'description': 'Available psychological resources',
                'column': 'resources',
            },
            {
                'description': 'Active stress intensity',
                'column': 'current_stress',
            },
            {
                'description': 'Perceived controllability of stress',
                'column': 'stress_controllability',
            },
            {
                'description': 'Perceived overload of stress events',
                'column': 'stress_overload',
            },
            {
                'description': 'Ongoing hindrance streak length',
                'column': 'consecutive_hindrances',
            }
        ]

        # Compute model data table
        model_rows = []
        for metric in model_metrics:
            try:
                if metric['agg_func'] == 'sum_per_agent':
                    if metric['column'] in model_df.columns:
                        series = model_df[metric['column']]
                    else:
                        logger.warning(f"Column {metric['column']} missing in model data.")
                        series = pd.Series(dtype=float)
                elif isinstance(metric['column'], list):
                    # For challenge vs hindrance
                    if all(col in model_df.columns for col in metric['column']):
                        challenge_stats = compute_stats(model_df['avg_challenge'])
                        hindrance_stats = compute_stats(model_df['avg_hindrance'])
                        mean_sd = f"{format_mean_sd(*challenge_stats[:2])} / {format_mean_sd(*hindrance_stats[:2])}"
                        min_max = format_min_max(min(challenge_stats[2], hindrance_stats[2]), max(challenge_stats[3], hindrance_stats[3]))
                    else:
                        logger.warning(f"Columns {metric['column']} missing in model data.")
                        mean_sd = "N/A"
                        min_max = "N/A"
                    interpretation = generate_interpretation(metric['description'], challenge_stats[0], challenge_stats[1], min(challenge_stats[2], hindrance_stats[2]), max(challenge_stats[3], hindrance_stats[3]))
                    model_rows.append({
                        'Description': metric['description'],
                        'Mean ± SD': mean_sd,
                        'Min - Max': min_max,
                        'Interpretation': interpretation
                    })
                    continue
                else:
                    if metric['column'] in model_df.columns:
                        series = model_df[metric['column']]
                    else:
                        logger.warning(f"Column {metric['column']} missing in model data.")
                        series = pd.Series(dtype=float)
                mean_val, std_val, min_val, max_val = compute_stats(series)
                interpretation = generate_interpretation(metric['description'], mean_val, std_val, min_val, max_val)
                model_rows.append({
                    'Description': metric['description'],
                    'Mean ± SD': format_mean_sd(mean_val, std_val),
                    'Min - Max': format_min_max(min_val, max_val),
                    'Interpretation': interpretation
                })
            except Exception as e:
                logger.warning(f"Error computing {metric['description']}: {e}")
                model_rows.append({
                    'Description': metric['description'],
                    'Mean ± SD': "N/A",
                    'Min - Max': "N/A",
                    'Interpretation': "Data unavailable"
                })

        model_df_table = pd.DataFrame(model_rows)
        model_path = os.path.join(output_dir, f"{prefix}_model_analysis.csv")
        model_df_table.to_csv(model_path, index=False)
        logger.info(f"Model analysis table saved to {model_path}")

        # Compute agent data table
        agent_rows = []
        for metric in agent_metrics:
            try:
                if metric['column'] in agent_df.columns:
                    series = agent_df[metric['column']]
                else:
                    logger.warning(f"Column {metric['column']} missing.")
                    series = pd.Series(dtype=float)
                mean_val, std_val, min_val, max_val = compute_stats(series)
                interpretation = generate_interpretation(metric['description'], mean_val, std_val, min_val, max_val)
                agent_rows.append({
                    'Description': metric['description'],
                    'Mean ± SD': format_mean_sd(mean_val, std_val),
                    'Min - Max': format_min_max(min_val, max_val),
                    'Interpretation': interpretation
                })
            except Exception as e:
                logger.warning(f"Error computing {metric['description']}: {e}")
                agent_rows.append({
                    'Description': metric['description'],
                    'Mean ± SD': "N/A",
                    'Min - Max': "N/A",
                    'Interpretation': "Data unavailable for interpretation"
                })

        agent_df_table = pd.DataFrame(agent_rows)
        agent_path = os.path.join(output_dir, f"{prefix}_agent_analysis.csv")
        agent_df_table.to_csv(agent_path, index=False)
        logger.info(f"Agent analysis table saved to {agent_path}")

        logger.info(f"Analysis completed successfully. Files saved to {output_dir}")

    except Exception as e:
        logger.warning(f"Analysis failed: {e}. Continuing execution.")
