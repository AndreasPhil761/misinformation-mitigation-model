import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scienceplots

plt.style.use(['science', 'ieee'])


def plot_opinion_set(ax, time_steps, user_opinions, recommender_opinion, color, linestyle, label=None, marker=None):

    # Handle different data structures
    if user_opinions.ndim == 3:
        # If 3D, take the first slice of the last dimension
        user_opinions = user_opinions[:, :, 0]

    # If the data has an extra time step, remove it
    if len(user_opinions) > len(time_steps):
        user_opinions = user_opinions[:-1]

    if len(recommender_opinion) > len(time_steps):
        recommender_opinion = recommender_opinion[:-1]

    n_users = user_opinions.shape[1]

    # Plot user opinions 
    for i in range(n_users):
        ax.plot(time_steps, user_opinions[:, i],
                color=color, linestyle=linestyle, linewidth=0.3, alpha=0.7,
                marker=marker, markersize=0.5, markeredgewidth=0.5)  # Reduced from 3 to 1.5

    # Plot recommender opinion 
    ax.plot(time_steps, recommender_opinion,
            color=color, linestyle=linestyle, linewidth=1.1, label=label,
            marker=marker, markersize=2.5, markeredgewidth=0.2)  # Reduced from 6 to 4


def add_steady_state_asymptotes(fig, datasets):

    # Get the axes object from the figure
    ax = fig.axes[0]  

    # Process each dataset
    for dataset in datasets:
        # Skip if no steady state is provided
        if 'steady_state' not in dataset or dataset['steady_state'] is None:
            continue

        steady_state = dataset['steady_state']
        steady_state = steady_state.flatten()
        color = dataset['color']
        label = dataset.get('label', None)

        # Add asymptotes for each steady state value
        if len(steady_state) > 0:
            # Add the first line with a label for the legend
            ax.axhline(y=steady_state[0], color=color, linestyle='dashdot',
                       alpha=0.6, linewidth=0.3,
                       label=f"steady state" if label else "Steady state")

            # Add the rest without labels
            for i in range(1, len(steady_state)):
                ax.axhline(y=steady_state[i], color=color, linestyle='dashdot',
                           alpha=0.6, linewidth=0.3)

    # Update the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')

    return fig

def fig_creation_with_no_avg(num_steps, datasets, figure_name, legend_title):
    """
    Create a graph with multiple opinion datasets

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

    # Plot each dataset
    for dataset in datasets:
        plot_opinion_set(
            ax=ax,
            time_steps=time_steps,
            user_opinions=dataset['user_opinions'],
            recommender_opinion=dataset['recommender_opinion'],
            color=dataset['color'],
            linestyle=dataset['linestyle'],
            label=dataset['label'],
            marker=dataset.get('marker', None)  # Get marker if provided, default to None
        )


    # Customize plot
    ax.set_xlabel('Time step')
    ax.set_ylabel('Extremity')
    ax.set_xlim(1, num_steps)
    ax.set_ylim(0, 1)

    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(which='major', length=4, width=0.8)
    ax.tick_params(which='minor', length=2, width=0.6)

    # Style spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Add legend
    legend = ax.legend(loc='best', frameon=True)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)

    add_steady_state_asymptotes(fig, datasets)
    # Save and show
    plt.tight_layout()
    fig.savefig(figure_name, dpi=1000, bbox_inches='tight')
    img = Image.open(figure_name)
    img.show()

    return fig