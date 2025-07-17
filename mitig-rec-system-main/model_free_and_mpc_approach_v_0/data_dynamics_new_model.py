import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from copy import deepcopy

def update_opinions(x, u, x0, A, B, lamda_diagonal_users):
    new_opinions = A @ x + B * u + lamda_diagonal_users @ x0
    return new_opinions


def simulate_dynamics_mf_data_new_model(x0, grouped_posts, num_steps, n_users, A, B, lamda_diagonal_users,
                                        a_const, lambda_val, range_window):
    x = x0
    opinion_history = [x]
    selected_posts = []
    available_posts = deepcopy(grouped_posts)  # Create working copy of posts

    for tstep in range(num_steps):
        # Determine valid time window for posts
        min_tc = max(0, tstep - range_window)
        max_tc = tstep

        # Get candidate posts within time window that haven't been selected
        candidate_posts = [
            post for post in available_posts
            if (min_tc <= post['tc'] <= max_tc) and ('selected' not in post)
        ]

        if not candidate_posts:
            print(f"No available posts at timestep {tstep} (new)")
            opinion_history.append(x)  # Maintain state if no selection
            continue

        sum_x_squared = np.sum(x ** 2)
        sum_x = np.sum(x)
        sum_ones = n_users
        #theoretical_max = np.linalg.norm(-1 * np.ones(n_users) - np.ones(n_users), ord=2)

        best_objective = float('inf')
        best_post = None

        for post in candidate_posts:
            u_temp = post['extremity']
            current_t_tc = tstep - post['tc']  # Dynamic t-tc calculation

            # Original alignment term
            original_theta_term = (sum_x_squared - 2 * u_temp * sum_x + u_temp ** 2 * sum_ones) #/ theoretical_max

            # Novelty+extremity term with dynamic t-tc
            en_term = a_const * np.square(np.abs(u_temp)) * np.exp(-lambda_val * current_t_tc)

            # Combined objective
            objective_value = original_theta_term + en_term

            if objective_value < best_objective:
                best_objective = objective_value
                best_post = post

        if best_post:
            # Mark post as selected and update opinions
            #print(f"selected post {tstep}: {best_post['content']}")
            best_post['selected'] = True
            u = best_post['extremity']
            x = update_opinions(x, u, x0, A, B, lamda_diagonal_users)
            selected_posts.append(best_post)

        opinion_history.append(x)

    # Calculate misinformation metrics
    u_values = np.array([post['extremity'] for post in selected_posts])
    labelled = np.array([post['label'] for post in selected_posts])

    total_false = sum(1 for post in grouped_posts if post['label'] == 'false')
    selected_false = np.sum(labelled == 'false')
    misinformation_spread = selected_false / (2 * total_false if total_false > 0 else 0)

    print("Finished data driven new MF")

    return np.array(opinion_history), u_values, misinformation_spread

def compute_convergence_values_mf_data_new(n_users, A, B, lamda_diagonal, initial_user_opinion, a):
    n = n_users
    identity = np.eye(n)
    s_mf = np.linalg.inv(identity - A - (1 / (n + a)) * B @ np.ones((1, n))) @ lamda_diagonal
    x_star_mf = s_mf @ initial_user_opinion
    return x_star_mf