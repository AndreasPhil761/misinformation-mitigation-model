import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from copy import deepcopy

def update_opinions(x, u, x0, A, B, lamda_diagonal_users):
    new_opinions = A @ x + B * u + lamda_diagonal_users @ x0
    return new_opinions


def simulate_old_mpc_dynamics(x0, grouped_posts, num_steps, n_users, A, B, lamda_diagonal_users, range_window,
                          future_prediction_range):
    x = x0
    opinion_history = [x]
    selected_posts = []
    available_posts = deepcopy(grouped_posts)  # Create a copy to track available posts

    # Calculate the theoretical maximum for normalization
    #theoretical_max = np.linalg.norm(-1 * np.ones(n_users) - np.ones(n_users), ord=2)

    for current_step in range(num_steps):
        # Determine the range of timesteps to consider for the current step
        min_tc = max(0, current_step - range_window)
        max_tc = current_step

        # Filter posts within the moving window and not yet selected
        candidate_posts = [
            post for post in available_posts
            if min_tc <= post['tc'] <= max_tc and 'selected' not in post
        ]

        if not candidate_posts:
            print(f"No available posts to select at timestep {current_step}.")
            continue

        # Perform MPC optimization
        best_total_objective = float('inf')
        best_first_post = None

        # Recursive function to explore future steps
        def explore_future_steps(current_x, step_index, prediction_horizon, selected_seq, objective_sum):
            # Base case: we've reached the end of our prediction horizon
            if step_index >= prediction_horizon:
                return objective_sum, selected_seq

            # If we've reached the original step count, still predict but don't go further
            if current_step + step_index >= num_steps:
                return objective_sum, selected_seq

            # Get candidate posts for this step in the prediction horizon
            future_min_tc = max(0, current_step + step_index - range_window)
            future_max_tc = current_step + step_index

            future_candidates = [
                post for post in available_posts
                if future_min_tc <= post['tc'] <= future_max_tc
                   and 'selected' not in post
                   and post not in selected_seq  # Ensure we don't reuse posts
            ]

            if not future_candidates:
                # If no posts available for this step, just return current results
                return objective_sum, selected_seq

            best_obj_from_here = float('inf')
            best_seq_from_here = None

            # Try each available post for this step
            for post in future_candidates:
                u_temp = post['extremity']

                # Calculate objective for this post
                sum_x_squared = np.sum(current_x ** 2)
                sum_x = np.sum(current_x)
                sum_ones = n_users
                step_objective = (sum_x_squared - 2 * u_temp * sum_x + u_temp ** 2 * sum_ones) #/ theoretical_max

                # Update opinions using this post
                new_x = update_opinions(current_x, u_temp, x0, A, B, lamda_diagonal_users)

                # Create new sequence with this post
                new_seq = selected_seq + [post]

                # Recursively explore the next step
                future_obj, future_seq = explore_future_steps(
                    new_x, step_index + 1, prediction_horizon, new_seq, objective_sum + step_objective
                )

                # Update best solution if this one is better
                if future_obj < best_obj_from_here:
                    best_obj_from_here = future_obj
                    best_seq_from_here = future_seq

            return best_obj_from_here, best_seq_from_here

        # For each first-step post, explore the future steps
        for first_post in candidate_posts:
            u_first = first_post['extremity']

            # Calculate objective for the first post
            sum_x_squared = np.sum(x ** 2)
            sum_x = np.sum(x)
            sum_ones = n_users
            first_objective = (sum_x_squared - 2 * u_first * sum_x + u_first ** 2 * sum_ones) #/ theoretical_max

            # Update opinions using the first post
            new_x = update_opinions(x, u_first, x0, A, B, lamda_diagonal_users)

            # Explore future steps starting from this first post
            total_objective, _ = explore_future_steps(
                new_x, 1, min(future_prediction_range, num_steps - current_step), [first_post], first_objective
            )

            # Update best solution if this one is better
            if total_objective < best_total_objective:
                best_total_objective = total_objective
                best_first_post = first_post

        if best_first_post is None:
            print(f"No valid sequence found at timestep {current_step}.")
            continue

        # Mark the selected post as "selected"
        best_first_post['selected'] = True

        # Update opinions using the selected post
        u = best_first_post['extremity']
        x = update_opinions(x, u, x0, A, B, lamda_diagonal_users)
        #print("Step\n")
        #print(u)
        #print(x)
        opinion_history.append(x)
        selected_posts.append(best_first_post)
        print("a timestep just finished (old MPC)")

    # Calculate misinformation spread metrics
    u_values = np.array([post['extremity'] for post in selected_posts])
    labelled = np.array([post['label'] for post in selected_posts])
    labelled_total = np.array([post['label'] for post in grouped_posts])
    selected_false_posts = np.sum(labelled == 'false')
    selected_labelled_total = np.sum(labelled_total == 'false')
    misinformation_spread = selected_false_posts / (2 * selected_labelled_total if selected_labelled_total > 0 else 0)

    print("Finished data driven old MPC")
    return np.array(opinion_history), u_values, misinformation_spread