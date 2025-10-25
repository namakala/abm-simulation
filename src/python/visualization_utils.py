"""
Visualization utilities for agent-based model analysis.

This module provides functions for generating comprehensive visualizations
of agent populations, including distribution plots, correlation analysis,
and statistical summaries.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
    # Set up matplotlib for consistent styling
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

# Optional scipy import for Q-Q plots
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None


def create_visualization_report(
    agents: List,
    output_dir: str = "outputs",
    filename: Optional[str] = None
) -> str:
    """
    Create comprehensive visualization report for population analysis.

    Args:
        agents: List of Person agents to analyze
        output_dir: Directory to save the visualization
        filename: Optional filename for saving the report

    Returns:
        Path to the generated report file, or placeholder if matplotlib not available
    """
    if not HAS_MATPLOTLIB:
        # Return placeholder path if matplotlib not available
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        placeholder_path = output_path / "visualization_not_available.txt"
        # Create the placeholder file if it doesn't exist
        if not placeholder_path.exists():
            with open(placeholder_path, 'w') as f:
                f.write("Matplotlib not available for visualization generation.\nInstall matplotlib to enable visualization features.")
        return str(placeholder_path)

    if filename is None:
        filename = f"population_analysis_{len(agents)}_agents.png"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Agent Population Analysis (N={len(agents)})', fontsize=16)

    # Extract data
    resilience_vals = [agent.resilience for agent in agents]
    affect_vals = [agent.affect for agent in agents]
    resources_vals = [agent.resources for agent in agents]
    pss10_vals = [agent.pss10 for agent in agents]

    # 1. Distribution plots with error handling
    try:
        axes[0, 0].hist(resilience_vals, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Resilience Distribution')
        axes[0, 0].set_xlabel('Resilience')
        axes[0, 0].set_ylabel('Frequency')
        mean_res = np.mean(resilience_vals)
        axes[0, 0].axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.3f}')
        axes[0, 0].legend()
    except (TypeError, ValueError):
        axes[0, 0].text(0.5, 0.5, 'Invalid resilience data', ha='center', va='center', transform=axes[0, 0].transAxes)

    try:
        axes[0, 1].hist(affect_vals, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Affect Distribution')
        axes[0, 1].set_xlabel('Affect')
        axes[0, 1].set_ylabel('Frequency')
        mean_aff = np.mean(affect_vals)
        axes[0, 1].axvline(mean_aff, color='red', linestyle='--', label=f'Mean: {mean_aff:.3f}')
        axes[0, 1].legend()
    except (TypeError, ValueError):
        axes[0, 1].text(0.5, 0.5, 'Invalid affect data', ha='center', va='center', transform=axes[0, 1].transAxes)

    try:
        axes[0, 2].hist(resources_vals, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Resources Distribution')
        axes[0, 2].set_xlabel('Resources')
        axes[0, 2].set_ylabel('Frequency')
        mean_res = np.mean(resources_vals)
        axes[0, 2].axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.3f}')
        axes[0, 2].legend()
    except (TypeError, ValueError):
        axes[0, 2].text(0.5, 0.5, 'Invalid resources data', ha='center', va='center', transform=axes[0, 2].transAxes)

    # 2. Box plots for comparison with error handling
    try:
        box_data = [resilience_vals, affect_vals, resources_vals]
        axes[1, 0].boxplot(box_data, labels=['Resilience', 'Affect', 'Resources'])
        axes[1, 0].set_title('Attribute Comparison')
        axes[1, 0].set_ylabel('Value')
    except (TypeError, ValueError):
        axes[1, 0].text(0.5, 0.5, 'Invalid boxplot data', ha='center', va='center', transform=axes[1, 0].transAxes)

    # 3. Scatter plots for relationships with error handling
    try:
        axes[1, 1].scatter(resilience_vals, affect_vals, alpha=0.6)
        axes[1, 1].set_xlabel('Resilience')
        axes[1, 1].set_ylabel('Affect')
        axes[1, 1].set_title('Resilience vs Affect')
        axes[1, 1].grid(True, alpha=0.3)
    except (TypeError, ValueError):
        axes[1, 1].text(0.5, 0.5, 'Invalid scatter data', ha='center', va='center', transform=axes[1, 1].transAxes)

    try:
        axes[1, 2].scatter(resilience_vals, resources_vals, alpha=0.6)
        axes[1, 2].set_xlabel('Resilience')
        axes[1, 2].set_ylabel('Resources')
        axes[1, 2].set_title('Resilience vs Resources')
        axes[1, 2].grid(True, alpha=0.3)
    except (TypeError, ValueError):
        axes[1, 2].text(0.5, 0.5, 'Invalid scatter data', ha='center', va='center', transform=axes[1, 2].transAxes)

    # 4. Q-Q plots for normality assessment with error handling
    if HAS_SCIPY:
        try:
            # Resilience Q-Q plot
            (osm, osr), (slope, intercept, r) = stats.probplot(resilience_vals, dist="norm")
            axes[2, 0].scatter(osm, osr)
            axes[2, 0].plot(osm, slope * osm + intercept, color='red', linestyle='--')
            axes[2, 0].set_xlabel('Theoretical Quantiles')
            axes[2, 0].set_ylabel('Sample Quantiles')
            axes[2, 0].set_title(f'Resilience Q-Q Plot (R²={r**2:.3f})')
            axes[2, 0].grid(True, alpha=0.3)
        except (TypeError, ValueError):
            axes[2, 0].text(0.5, 0.5, 'Invalid Q-Q plot data', ha='center', va='center', transform=axes[2, 0].transAxes)
    else:
        axes[2, 0].text(0.5, 0.5, 'SciPy not available\nfor Q-Q plot',
                       ha='center', va='center', transform=axes[2, 0].transAxes)

    # 5. Correlation heatmap with error handling
    try:
        correlation_data = np.array([resilience_vals, affect_vals, resources_vals])
        corr_matrix = np.corrcoef(correlation_data)

        sns.heatmap(corr_matrix,
                    annot=True,
                    cmap='coolwarm',
                    vmin=-1, vmax=1,
                    xticklabels=['Resilience', 'Affect', 'Resources'],
                    yticklabels=['Resilience', 'Affect', 'Resources'],
                    ax=axes[2, 1])
        axes[2, 1].set_title('Correlation Matrix')
    except (TypeError, ValueError):
        axes[2, 1].text(0.5, 0.5, 'Invalid correlation data', ha='center', va='center', transform=axes[2, 1].transAxes)

    # 6. PSS-10 distribution with error handling
    try:
        axes[2, 2].hist(pss10_vals, bins=range(0, 41, 2), alpha=0.7, edgecolor='black')
        axes[2, 2].set_title('PSS-10 Score Distribution')
        axes[2, 2].set_xlabel('PSS-10 Score')
        axes[2, 2].set_ylabel('Frequency')
        mean_pss = np.mean(pss10_vals)
        axes[2, 2].axvline(mean_pss, color='red', linestyle='--', label=f'Mean: {mean_pss:.1f}')
        axes[2, 2].legend()
    except (TypeError, ValueError):
        axes[2, 2].text(0.5, 0.5, 'Invalid PSS-10 data', ha='center', va='center', transform=axes[2, 2].transAxes)

    plt.tight_layout()

    # Save the report
    report_path = output_path / filename
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(report_path)