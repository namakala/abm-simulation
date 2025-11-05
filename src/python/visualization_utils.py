"""
Visualization utilities for agent-based model analysis.

This module provides functions for generating comprehensive visualizations
of agent populations, including distribution plots, correlation analysis,
and statistical summaries.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
    # Set up matplotlib for consistent styling
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None
    GridSpec = None

# Optional scipy import for Q-Q plots and correlations
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None


def create_visualization_report(
    data: pd.DataFrame,
    output_dir: str = "outputs",
    filename: Optional[str] = None,
    style_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create comprehensive visualization report for psychological data analysis.

    Args:
        data: DataFrame with columns for resilience, affect, stress, and PSS-10 scores
        output_dir: Directory to save the visualization
        filename: Optional filename for saving the report (defaults to PDF)
        style_config: Optional dictionary for customizing plot styling

    Returns:
        Path to the generated PDF report file, or placeholder if matplotlib not available
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

    # Set up PDF backend for high-resolution output
    matplotlib.use('pdf')

    # Default style configuration
    if style_config is None:
        style_config = {
            'figure_size': (20, 10),
            'dpi': 300,
            'palette': 'husl',
            'font_size': 10,
            'title_font_size': 14
        }

    # Apply style configuration
    plt.style.use('default')
    sns.set_palette(style_config.get('palette', 'husl'))
    plt.rcParams['font.size'] = style_config.get('font_size', 12)
    plt.rcParams['axes.titlesize'] = style_config.get('title_font_size', 16)

    if filename is None:
        filename = f"psychological_data_report_{len(data)}_observations.pdf"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Handle missing data
    data = data.dropna(subset=['resilience', 'affect', 'stress', 'pss10'])
    if data.empty:
        raise ValueError("No valid data after removing missing values")

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=style_config.get('figure_size', (20, 10)))
    gs = GridSpec(2, 12, figure=fig, hspace=0.6, wspace=0.6)
    fig.suptitle(f'Psychological Data Analysis Report (N={len(data)})', fontsize=style_config.get('title_font_size', 14), y=0.98)

    # Extract data
    resilience_vals = data['resilience'].values
    affect_vals = data['affect'].values
    stress_vals = data['stress'].values
    pss10_vals = data['pss10'].values

    # 1. Distribution plots with KDE and stats
    variables = [
        ('resilience', resilience_vals, 'Resilience'),
        ('affect', affect_vals, 'Affect'),
        ('stress', stress_vals, 'Stress'),
        ('pss10', pss10_vals, 'PSS-10')
    ]

    for i, (col, vals, title) in enumerate(variables):
        try:
            ax = fig.add_subplot(gs[0, i*3:(i+1)*3])
            # Histogram with KDE
            sns.histplot(vals, bins=30, kde=True, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'{title} Distribution', fontsize=style_config.get('font_size', 10) + 2)
            ax.set_xlabel(title, fontsize=style_config.get('font_size', 10))
            ax.set_ylabel('Frequency', fontsize=style_config.get('font_size', 10))
            ax.tick_params(axis='both', which='major', labelsize=style_config.get('font_size', 10) - 2)

            # Set x-axis limits based on variable
            if col == 'affect':
                ax.set_xlim(-1, 1)
            elif col == 'resilience':
                ax.set_xlim(0, 1)
            elif col == 'stress':
                ax.set_xlim(0, 1)
            elif col == 'pss10':
                ax.set_xlim(0, 40)

            # Add statistics
            mean_val = np.mean(vals)
            median_val = np.median(vals)
            std_val = np.std(vals)

            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='blue', linestyle=':', label=f'Median: {median_val:.3f}')

            # Add stats text
            stats_text = f'σ: {std_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=style_config.get('font_size', 10) - 2, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.legend(fontsize=style_config.get('font_size', 10) - 2)
        except (TypeError, ValueError):
            ax = fig.add_subplot(gs[0, i*3:(i+1)*3])
            ax.text(0.5, 0.5, f'Invalid {col} data', ha='center', va='center', transform=ax.transAxes, fontsize=style_config.get('font_size', 10))

    # 2. Specific 2D scatter plots with regression equations
    scatter_pairs = [
        ('resilience', 'affect', 'Resilience vs Affect'),
        ('resilience', 'stress', 'Resilience vs Stress'),
        ('resilience', 'pss10', 'Resilience vs PSS-10'),
        ('pss10', 'stress', 'PSS-10 vs Stress')
    ]

    for i, (x_col, y_col, title) in enumerate(scatter_pairs):
        try:
            ax = fig.add_subplot(gs[1, i*3:(i+1)*3])
            x_vals = data[x_col].values
            y_vals = data[y_col].values

            # Scatter plot with regression line
            sns.regplot(x=x_vals, y=y_vals, ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})

            ax.set_xlabel(x_col.capitalize(), fontsize=style_config.get('font_size', 10))
            ax.set_ylabel(y_col.capitalize(), fontsize=style_config.get('font_size', 10))
            ax.set_title(title, fontsize=style_config.get('font_size', 10) + 2)
            ax.tick_params(axis='both', which='major', labelsize=style_config.get('font_size', 10) - 2)
            ax.grid(True, alpha=0.3)

            # Add regression equation and correlation coefficient
            if HAS_SCIPY:
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                corr = r_value

                # Format regression equation
                eq_text = f'y = {slope:.3f}x + {intercept:.3f}'
                corr_text = f'r = {corr:.3f}, p = {p_value:.2e}'

                # Combine texts
                full_text = f'{eq_text}\n{corr_text}'
                ax.text(0.02, 0.98, full_text, transform=ax.transAxes,
                               verticalalignment='top', fontsize=style_config.get('font_size', 10) - 2, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except (TypeError, ValueError):
            ax = fig.add_subplot(gs[1, i*3:(i+1)*3])
            ax.text(0.5, 0.5, f'Invalid {x_col} vs {y_col} data', ha='center', va='center', transform=ax.transAxes, fontsize=style_config.get('font_size', 10))


    # Adjust layout and save
    # Note: tight_layout() commented out to avoid warning with legend positioning
    # plt.tight_layout()

    # Save the report as high-resolution PDF
    report_path = output_path / filename
    plt.savefig(report_path, dpi=style_config.get('dpi', 300), bbox_inches='tight', format='pdf')
    plt.close()

    return str(report_path)


def create_time_series_visualization(
    model_data: pd.DataFrame,
    output_dir: str = "docs/figures",
    filename: str = "time_series.pdf"
) -> str:
    """
    Create time series visualization with z-score normalized averages and moving averages.

    Args:
        model_data: DataFrame with time series data containing columns:
                   'avg_pss10', 'avg_stress', 'avg_resilience', 'avg_affect'
        output_dir: Directory to save the visualization
        filename: Output filename (default: 'time_series.pdf')

    Returns:
        Path to the generated PDF file, or placeholder if matplotlib not available
    """
    if not HAS_MATPLOTLIB:
        # Return placeholder path if matplotlib not available
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        placeholder_path = output_path / "time_series_not_available.txt"
        # Create the placeholder file if it doesn't exist
        if not placeholder_path.exists():
            with open(placeholder_path, 'w') as f:
                f.write("Matplotlib not available for time series visualization generation.\nInstall matplotlib to enable visualization features.")
        return str(placeholder_path)

    # Set up PDF backend for high-resolution output
    matplotlib.use('pdf')

    # Default style configuration
    style_config = {
        'figure_size': (16, 10),
        'dpi': 300,
        'font_size': 10,
        'title_font_size': 12
    }

    # Apply style configuration
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['font.size'] = style_config.get('font_size', 12)
    plt.rcParams['axes.titlesize'] = style_config.get('title_font_size', 16)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate required columns
    required_cols = ['avg_pss10', 'avg_stress', 'avg_resilience', 'avg_affect']
    missing_cols = [col for col in required_cols if col not in model_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter data to required columns and drop NaN
    plot_data = model_data[required_cols].dropna()
    if plot_data.empty:
        raise ValueError("No valid data after removing missing values")

    # Create figure with GridSpec for 2x2 layout
    fig = plt.figure(figsize=style_config.get('figure_size', (16, 10)))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4, top=0.85)
    fig.suptitle('Time Series Analysis: Mental Health Metrics', fontsize=style_config.get('title_font_size', 14), y=0.95)

    # Create 2x2 subplots
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]

    # Define variables and their labels in 2x2 order: TL PSS-10, TR Stress, BL Resilience, BR Affect
    variables = [
        ('avg_pss10', 'PSS-10'),
        ('avg_stress', 'Stress'),
        ('avg_resilience', 'Resilience'),
        ('avg_affect', 'Affect')
    ]

    # Moving averages configuration
    moving_windows = [7, 14, 28]
    ma_colors = ['blue', 'green', 'red']
    alphas = [0.2, 0.4, 0.6, 0.8]  # raw, 7-step, 14-step, 28-step

    # Plot each variable in its subplot
    for idx, (col, label) in enumerate(variables):
        row = idx // 2
        col_idx = idx % 2
        ax = axes[row][col_idx]

        # Plot raw data
        ax.plot(plot_data.index, plot_data[col], 'k-', alpha=alphas[0], linewidth=1, label='Raw')

        # Plot moving averages
        for window, color, alpha in zip(moving_windows, ma_colors, alphas[1:]):
            if len(plot_data) >= window:
                ma = plot_data[col].rolling(window=window, center=True).mean()
                ax.plot(plot_data.index, ma, color=color, linewidth=2,
                       label=f'{window}-step MA', alpha=alpha)

        ax.set_title(label, fontsize=style_config.get('font_size', 10) + 2)
        ax.set_xlabel('Time Step', fontsize=style_config.get('font_size', 10))
        ax.set_ylabel(label, fontsize=style_config.get('font_size', 10))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=style_config.get('font_size', 10) - 2)

    # Add single legend below title, above subplots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=4,
               fontsize=style_config.get('font_size', 10) - 2)

    # Adjust layout and save
    plt.tight_layout()

    # Save the visualization as high-resolution PDF
    viz_path = output_path / filename
    plt.savefig(viz_path, dpi=style_config.get('dpi', 300), bbox_inches='tight', format='pdf')
    plt.close()

    return str(viz_path)
