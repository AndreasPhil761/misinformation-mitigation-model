import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
#import cvxpy as cp

def update_opinions(x, u, x0, A, B, lamda_diagonal_users):
    new_opinions = A @ x + B * u + lamda_diagonal_users @ x0
    return new_opinions

def simulate_dynamics_mf_data_old_model(x0, grouped_posts, num_steps, n_users, A, B, lamda_diagonal_users, range_window):
    x = x0
    opinion_history = [x]
    selected_posts = []
    available_posts = grouped_posts.copy()  # Create a copy to track available posts

    for tstep in range(num_steps):
        # Determine the range of timesteps to consider
        min_tc = max(0, tstep - range_window)
        max_tc = tstep

        # Filter posts within the moving window and not yet selected
        candidate_posts = [
            post for post in available_posts
            if min_tc <= post['tc'] <= max_tc and 'selected' not in post
        ]

        if not candidate_posts:
            print(f"No available posts to select at timestep {tstep} (old).")
            continue

        sum_x_squared = np.sum(x ** 2)
        sum_x = np.sum(x)
        sum_ones = n_users
        #theoretical_max = np.linalg.norm(-1 * np.ones(n_users) - np.ones(n_users), ord=2)

        best_objective = float('inf')
        best_post = None

        # Select the best post based on the objective function
        for post in candidate_posts:
            u_temp = post['extremity']
            original_theta_term = (sum_x_squared - 2 * u_temp * sum_x + u_temp ** 2 * sum_ones) #/ theoretical_max
            objective_value = original_theta_term
            if objective_value < best_objective:
                best_objective = objective_value
                best_post = post

        # Mark the selected post as "selected" by adding a flag
        best_post['selected'] = True

        # Update opinions using the selected post
        u = best_post['extremity']
        x = update_opinions(x, u, x0, A, B, lamda_diagonal_users)
        opinion_history.append(x)
        selected_posts.append(best_post)

    u_values = np.array([post['extremity'] for post in selected_posts])
    labelled = np.array([post['label'] for post in selected_posts])
    labelled_total = np.array([post['label'] for post in grouped_posts])
    selected_false_posts = np.sum(labelled == 'false')
    selected_labelled_total = np.sum(labelled_total == 'false')
    misinformation_spread = selected_false_posts / (2 * selected_labelled_total)
    print("Finished data driven old MF")
    return np.array(opinion_history), u_values, misinformation_spread