import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scienceplots

plt.style.use(['science', 'ieee'])


def plot_opinion_set(ax, time_steps, user_opinions, recommender_opinion, color, linestyle, label=None, marker=None):
    """
    Plot mean user opinion with confidence bands and recommender opinion
    Returns handles for custom legend creation
    """
    # Handle different data structures
    if user_opinions.ndim == 3:
        # If 3D, take the first slice of the last dimension
        user_opinions = user_opinions[:, :, 0]

    # If the data has an extra time step, remove it
    if len(user_opinions) > len(time_steps):
        user_opinions = user_opinions[:-1]

    if len(recommender_opinion) > len(time_steps):
        recommender_opinion = recommender_opinion[:-1]

    # Calculate mean and standard deviation across users (axis=1)
    mean_opinion = np.mean(user_opinions, axis=1)
    std_opinion = np.std(user_opinions, axis=1)
    
    # Plot mean user opinion (no label - will be handled in custom legend)
    mean_line = ax.plot(time_steps, mean_opinion, 
                       color=color, linestyle=':', linewidth=0.5, alpha=0.8)[0]
    
    # Add confidence band (no label - will be combined with mean line)
    fill_patch = ax.fill_between(time_steps, 
                                mean_opinion - std_opinion, 
                                mean_opinion + std_opinion,
                                color=color, alpha=0.12)

    # Plot recommender opinion (thick line)
    rec_line = ax.plot(time_steps, recommender_opinion,
                      color=color, linestyle=linestyle, linewidth=1.6, 
                      marker=marker, markersize=4.2, markeredgewidth=0.4)[0]
    
    # Return handles for custom legend
    return {
        'mean_line': mean_line,
        'fill_patch': fill_patch, 
        'rec_line': rec_line,
        'label': label
    }


def add_steady_state_asymptotes(fig, datasets):
    # Get the axes object from the figure
    ax = fig.axes[0]  # Assuming there's only one subplot

    steady_state_handles = []
    
    # Process each dataset
    for i, dataset in enumerate(datasets):
        # Skip if no steady state is provided
        if 'steady_state' not in dataset or dataset['steady_state'] is None:
            continue

        steady_state = dataset['steady_state']
        steady_state = steady_state.flatten()
        color = dataset['color']

        # Add asymptotes for each steady state value
        if len(steady_state) > 0:
            # Add the first line and store handle for legend
            line = ax.axhline(y=steady_state[0], color=color, linestyle='dashdot',
                             alpha=0.6, linewidth=0.3)
            
            # Only add to legend for the first dataset to avoid duplicate labels
            if i == 0:
                steady_state_handles.append((line, "Steady state"))

            # Add the rest without storing handles
            for j in range(1, len(steady_state)):
                ax.axhline(y=steady_state[j], color=color, linestyle='dashdot',
                          alpha=0.6, linewidth=0.3)

    return steady_state_handles


def fig_creation(num_steps, datasets, figure_name, legend_title):
    """
    Create a graph with multiple opinion datasets with custom overlayed legend

    Args:
        num_steps: number of time steps
        datasets: list of dictionaries, each containing:
                 'user_opinions': 2D array (time x users)
                 'recommender_opinion': 1D array (time)
                 'color': color for this dataset
                 'linestyle': line style or 'None'
                 'marker': marker style or None
                 'label': legend label
        figure_name: output filename
        legend_title: title for the legend
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))

    # Create time array
    time_steps = np.arange(1, num_steps + 1)

    # Store handles for custom legend
    legend_handles = []
    legend_labels = []

    # Plot each dataset and collect handles
    for dataset in datasets:
        handles = plot_opinion_set(
            ax=ax,
            time_steps=time_steps,
            user_opinions=dataset['user_opinions'],
            recommender_opinion=dataset['recommender_opinion'],
            color=dataset['color'],
            linestyle=dataset['linestyle'],
            label=dataset['label'],
            marker=dataset.get('marker', None)
        )
        
        # Create combined handle for users (fill + line overlayed)
        combined_users = (handles['fill_patch'], handles['mean_line'])
        legend_handles.append(combined_users)
        legend_labels.append(f"{handles['label']}usr")
        
        # Add recommender handle separately
        legend_handles.append(handles['rec_line'])
        legend_labels.append(f"{handles['label']}rec-out")

    # Add steady state lines to legend
    steady_state_handles = add_steady_state_asymptotes(fig, datasets)
    for handle, label in steady_state_handles:
        legend_handles.append(handle)
        legend_labels.append(label)

    # Customize plot
    ax.set_xlabel('Time-step', fontsize='large')
    ax.set_ylabel('Negative Emotional Extremity of Opinion', fontsize='large')
    ax.set_xlim(1, num_steps)
    ax.set_ylim(0, 1)

    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(which='major', length=4, width=0.8)
    ax.tick_params(which='minor', length=2, width=0.6)

    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    
    legend = ax.legend(legend_handles, legend_labels, 
                      loc='upper center',           
                      bbox_to_anchor=(0.5, -0.085), 
                      ncol=3,                       
                      frameon=False,                
                      fontsize='large',
                      columnspacing=0.5)            
    
    # Save and show
    plt.tight_layout()
    
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(figure_name, dpi=1000, bbox_inches='tight')
    img = Image.open(figure_name)
    img.show()

    return fig