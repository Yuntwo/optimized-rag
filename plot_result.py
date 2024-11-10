# _*_ coding: utf-8 _*_
__author__ = 'yuntwo'
__date__ = '2024/11/10 20:08:26'

import matplotlib.pyplot as plt
import numpy as np

# Baseline
metrics = ["Overall Hit Rate", "Direct Query Hit Rate", "Relevant Query Hit Rate"]
fulltext_values = [0.59, 0.660, 0.500]  # Fulltext search values for each metric
vector_values = [0.42, 0.232, 0.659]  # Vector search values for each metric
hybrid_values = [0.56, 0.482, 0.659]  # Hybrid search values for each metric


def plot(save_path="plot/baseline.png"):
    """
    Plots a comparison bar chart for Fulltext, Vector, and Hybrid search performance on specified metrics.

    Args:
    - save_path (str): File path to save the plot image (default is 'visualization/comparison_metrics.png')
    """
    # Define bar width and positions
    bar_width = 0.25
    x = np.arange(len(metrics))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    bars1 = ax.bar(x - bar_width, fulltext_values, width=bar_width, label='Fulltext Search', color='skyblue')
    bars2 = ax.bar(x, vector_values, width=bar_width, label='Vector Search', color='salmon')
    bars3 = ax.bar(x + bar_width, hybrid_values, width=bar_width, label='Hybrid Search (50% Fulltext, 50% Vector)', color='lightgreen')

    # Set y-axis limit from 0 to 1
    ax.set_ylim(0, 1)

    # Labeling and titles
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Fulltext, Vector, and Hybrid Search on Key Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    # Place the legend outside of the plot area
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Adjust layout to provide more space at the bottom for the legend
    plt.subplots_adjust(bottom=0.3)

    # Display values on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

    # Save and show plot
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as '{save_path}'.")

# Call the function to plot and save the chart
plot()
