import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cProfile import label

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import scienceplots

# Enable SciencePlots with IEEE style
plt.style.use(['science', 'ieee'])

def plot(n_users, time_steps, opinion_history, u_history, chosen_color, ax, style, lamda_val, text):
    # Plot opinions
    for i in range(n_users):
        ax.plot(time_steps, opinion_history[:-1, i, 0],
                color=chosen_color, linestyle=style, linewidth=0.3)

        # Add dashed horizontal line for theoretical max
        # ax.axhline(y=x_star_mf[i], color=orange_color,
        # linestyle='--', linewidth=0.4)

    # Plot recommender system opinion in a different color
    ax.plot(time_steps, u_history, color=chosen_color, linestyle=style,
                linewidth=1.1, label =text)

def make_graph(num_steps, n_users, opinion_history, u_history, opinion_histories_new, u_histories_new, opinion_history_data_mpc_old, u_history_data_mpc_old, u_histories_data_new_mpc, opinion_histories_data_new_mpc, lambda_val, name_of_file):
    # Create figure with square aspect ratio
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot user opinions (all in the same color)
    #opinion_color = 'orange'
    blue_color = '#0072B2'
    orange_color = '#E69F00'
    red_color = '#FF0000'
    black_color = 'black'
    green_color = 'green'
    # Create time array
    time_steps = np.arange(num_steps)

    plot(n_users,time_steps, opinion_history, u_history, orange_color, ax,'--', "",'old model mf')
    plot(n_users, time_steps, opinion_histories_new[1], u_histories_new[1], blue_color, ax, '--', lambda_val[0],'new model mf')
    plot(n_users,time_steps, opinion_history_data_mpc_old, u_history_data_mpc_old, red_color, ax,'--', "",'old model mpc')
    plot(n_users, time_steps, opinion_histories_data_new_mpc[1], u_histories_data_new_mpc[1], black_color, ax, '--', lambda_val[0], 'new model mpc')
    #plot(n_users, time_steps, opinion_histories_new[1], u_histories_new[1], red_color, ax, '--', lambda_val[1])
    #plot(n_users, time_steps, opinion_histories_new[2], u_histories_new[2], black_color, ax, '--', lambda_val[2])
    #plot(n_users, time_steps, opinion_histories_new[3], u_histories_new[3], green_color, ax, '--', lambda_val[3])
    #for i in range(len(lambda_val)):
        #plot(n_users, time_steps, opinion_histories_new[i], u_histories_new[i], blue_color, ax, '--')

    legend = ax.legend(loc='best', frameon=False, framealpha=1.0)
    #legend.get_frame().set_linewidth(0.5)

    # Set axis labels
    ax.set_xlabel('Time step')
    ax.set_ylabel('Opinion')

    # Set tight boundaries
    ax.set_xlim(0, num_steps-1)

    ax.set_ylim(0, 1)

    # Add minor ticks
    ax.minorticks_on()

    # Adjust tick parameters for IEEE style
    ax.tick_params(which='major', length=4, width=0.8)
    ax.tick_params(which='minor', length=2, width=0.6)

    # Ensure all spines are visible
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Tight layout to maximize plot area
    plt.tight_layout()

    fig.savefig(name_of_file, dpi=1000, bbox_inches='tight')
    img = Image.open(name_of_file)
    img.show()
    return fig