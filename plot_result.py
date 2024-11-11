import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Baseline and Approach Data
metrics = ["Direct Query Hit Rate", "Relevant Query Hit Rate", "Overall Hit Rate"]
fulltext_values = [0.660, 0.500, 0.59]  # Fulltext search values for each metric (baseline)
vector_values = [0.232, 0.659, 0.42]  # Vector search values for each metric (baseline)
hybrid_values = [0.482, 0.659, 0.56]  # Hybrid search values for each metric (baseline)

# Rewriting
# fulltext_approach_values = [0.464, 0.522, 0.49]
# vector_approach_values = [0.214, 0.682, 0.42]
# hybrid_approach_values = [0.321, 0.591, 0.44]

# Reranking
# fulltext_approach_values = [0.786, 0.500, 0.66]
# vector_approach_values = [0.036, 0.659, 0.31]
# hybrid_approach_values = [0.732, 0.682, 0.71]

# Rewriting + Reranking
fulltext_approach_values = [0.589, 0.523, 0.56]
vector_approach_values = [0.018, 0.705, 0.32]
hybrid_approach_values = [0.482, 0.591, 0.53]


def plot_baseline(save_path="plot/baseline.png"):
    """
    Plots a comparison bar chart for Fulltext, Vector, and Hybrid search performance on specified metrics.

    Args:
    - save_path (str): File path to save the plot_baseline image (default is 'plot_baseline/baseline.png')
    """
    # Define bar width and positions
    bar_width = 0.25
    x = np.arange(len(metrics))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    bars1 = ax.bar(x - bar_width, fulltext_values, width=bar_width, label='Fulltext Search', color='skyblue')
    bars2 = ax.bar(x, vector_values, width=bar_width, label='Vector Search', color='salmon')
    bars3 = ax.bar(x + bar_width, hybrid_values, width=bar_width, label='Hybrid Search (50% Fulltext, 50% Vector)',
                   color='lightgreen')

    # Set y-axis limit from 0 to 1
    ax.set_ylim(0, 1)

    # Labeling and titles
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Fulltext, Vector, and Hybrid Search on Key Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    # Place the legend outside of the plot_baseline area
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Adjust layout to provide more space at the bottom for the legend
    plt.subplots_adjust(bottom=0.3)

    # Display values on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')

    # Save and show plot_baseline
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as '{save_path}'.")


def plot_approach(save_path="plot/approach.png"):
    """
    Plots a comparison bar chart for Fulltext, Vector, and Hybrid search performance with and without the improvement method.

    Args:
    - save_path (str): File path to save the plot image (default is 'plot/approach.png')
    """
    # Define bar width and positions
    bar_width = 0.15
    x = np.arange(len(metrics))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Baseline bars
    bars1 = ax.bar(x - bar_width * 2, fulltext_values, width=bar_width, label='Baseline Fulltext', color='skyblue')
    bars2 = ax.bar(x, vector_values, width=bar_width, label='Baseline Vector', color='salmon')
    bars3 = ax.bar(x + bar_width * 2, hybrid_values, width=bar_width, label='Baseline Hybrid', color='lightgreen')

    # Approach bars with deeper colors and patterns
    bars4 = ax.bar(x - bar_width * 2 + bar_width, fulltext_approach_values, width=bar_width,
                   label='Approach Fulltext', color='royalblue', hatch='//')
    bars5 = ax.bar(x + bar_width, vector_approach_values, width=bar_width, label='Approach Vector',
                   color='darkred', hatch='//')
    bars6 = ax.bar(x + bar_width * 2 + bar_width, hybrid_approach_values, width=bar_width, label='Approach Hybrid',
                   color='darkgreen', hatch='//')

    # Set y-axis limit from 0 to 1
    ax.set_ylim(0, 1)

    # Labeling and titles
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Baseline and Approach for Fulltext, Vector, and Hybrid Search on Key Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    # Custom legend in two rows and three columns, with each pair in one column
    baseline_fulltext_patch = mpatches.Patch(color='skyblue', label='Baseline Fulltext')
    approach_fulltext_patch = mpatches.Patch(color='royalblue', label='Approach Fulltext', hatch='//')
    baseline_vector_patch = mpatches.Patch(color='salmon', label='Baseline Vector')
    approach_vector_patch = mpatches.Patch(color='darkred', label='Approach Vector', hatch='//')
    baseline_hybrid_patch = mpatches.Patch(color='lightgreen', label='Baseline Hybrid')
    approach_hybrid_patch = mpatches.Patch(color='darkgreen', label='Approach Hybrid', hatch='//')

    ax.legend(handles=[baseline_fulltext_patch, approach_fulltext_patch,
                       baseline_vector_patch, approach_vector_patch,
                       baseline_hybrid_patch, approach_hybrid_patch],
              loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Adjust layout to provide more space at the bottom for the legend
    plt.subplots_adjust(bottom=0.4)

    # Display values on top of bars for both baseline and approach
    for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')

    # Save and show plot
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as '{save_path}'.")


dimension = [1536, 768, 384, 100, 30]
overall_hit_rate = [0.42, 0.42, 0.41, 0.32, 0.1]


def plot_dimension_vs_hit_rate(dimension, overall_hit_rate, save_path="plot/vector_search_dim_reduction.png"):
    """
    Plots a scatter chart for Vector Search dimensionality reduction approach with dimensions on x-axis
    and overall hit rate on y-axis, with the x-axis arranged from largest to smallest values.

    Args:
    - dimension (list): List of dimensions for the x-axis.
    - overall_hit_rate (list): List of overall hit rates corresponding to each dimension.
    - save_path (str): File path to save the plot image.
    """
    # Reverse the data to make x-axis display from large to small
    dimension = dimension[::-1]
    overall_hit_rate = overall_hit_rate[::-1]

    plt.figure(figsize=(10, 6))
    plt.scatter(dimension, overall_hit_rate, color='b', label='Overall Hit Rate')

    # Add labels and title
    plt.xlabel('Dimension')
    plt.ylabel('Overall Hit Rate')
    plt.title('Vector Search Dimensionality Reduction Approach')

    # Connect points with a line
    plt.plot(dimension, overall_hit_rate, color='b', linestyle='--', linewidth=0.8)

    # Annotate points and display dimension values
    for x, y in zip(dimension, overall_hit_rate):
        plt.text(x, y + 0.02, f'{y:.2f}', ha='center', va='bottom')  # Hit Rate above the point
        plt.text(x, y - 0.04, f'{x}', ha='center', va='top')          # Dimension below the point

    # Set y-axis to log scale for better readability if dimensions vary widely
    plt.xscale('log')
    plt.gca().invert_xaxis()  # Invert x-axis to display from large to small

    # Adjust y-axis limit to provide extra space at the top
    plt.ylim(0, 0.6)

    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()
    print(f"Plot saved as '{save_path}'.")


# Call the functions to plot and save the charts
# plot_baseline()
# plot_approach()
plot_dimension_vs_hit_rate(dimension=dimension, overall_hit_rate=overall_hit_rate)
