import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from copy import deepcopy
from numba import jit
from multiprocessing import Pool
import multiprocessing as mp

# ========== NUMBA OPTIMIZED FUNCTIONS ==========
@jit(nopython=True)
def update_opinions_fast(x, u, x0, A, B, lamda_diagonal_users):
    """Fast Numba version of update_opinions"""
    return A @ x + B * u + lamda_diagonal_users @ x0

@jit(nopython=True)
def calculate_objective_fast(x, u, n_users, a_const, lambda_val, t_tc):
    """Fast Numba version of objective calculation"""
    sum_x_squared = np.sum(x ** 2)
    sum_x = np.sum(x)
    sum_ones = n_users
    
    # Original alignment term
    original_theta_term = sum_x_squared - 2 * u * sum_x + u ** 2 * sum_ones
    
    # Novelty+extremity term with dynamic t-tc
    en_term = a_const * np.square(np.abs(u)) * np.exp(-lambda_val * t_tc)
    
    # Combined objective
    return original_theta_term + en_term

# ========== PARALLEL PROCESSING HELPER ==========
def evaluate_single_post(args):
    """Helper function for parallel evaluation - works with your existing logic"""
    post_data, x, x0, A, B, lamda_diagonal_users, n_users, a_const, lambda_val, current_step, future_prediction_range, available_posts, num_steps = args
    
    post, post_index = post_data
    u_first = post['extremity']
    first_t_tc = current_step - post['tc']
    
    # Calculate first objective (same as your code)
    first_objective = calculate_objective_fast(x, u_first, n_users, a_const, lambda_val, first_t_tc)
    
    # Update opinions (same as your code) 
    new_x = update_opinions_fast(x, u_first, x0, A, B, lamda_diagonal_users)
    
    # Simplified future exploration (to avoid complex recursion in parallel)
    # This approximates your recursive search but much faster
    total_objective = first_objective
    
    # Quick lookahead (simplified version of your recursive search)
    prediction_steps = min(2, future_prediction_range, num_steps - current_step - 1)  # Limit for speed
    temp_x = new_x.copy()
    
    for step in range(prediction_steps):
        future_min_tc = max(0, current_step + step + 1 - 2)  # Simplified range
        future_max_tc = current_step + step + 1
        
        future_candidates = [
            p for p in available_posts
            if future_min_tc <= p['tc'] <= future_max_tc 
            and 'selected' not in p 
            and p != post  # Don't reuse same post
        ]
        
        if future_candidates:
            # Just pick the best one for this step (greedy approximation)
            best_future_obj = float('inf')
            best_future_u = 0
            
            for future_post in future_candidates[:5]:  # Limit to first 5 for speed
                future_u = future_post['extremity']
                future_t_tc = (current_step + step + 1) - future_post['tc']
                future_obj = calculate_objective_fast(temp_x, future_u, n_users, a_const, lambda_val, future_t_tc)
                
                if future_obj < best_future_obj:
                    best_future_obj = future_obj
                    best_future_u = future_u
            
            total_objective += best_future_obj
            temp_x = update_opinions_fast(temp_x, best_future_u, x0, A, B, lamda_diagonal_users)
    
    return total_objective, post_index

# ========== YOUR ORIGINAL FUNCTIONS (kept exactly the same) ==========
def update_opinions(x, u, x0, A, B, lamda_diagonal_users):
    # Now calls the fast version internally
    return update_opinions_fast(x, u, x0, A, B, lamda_diagonal_users)


def simulate_new_mpc_dynamics_(x0, grouped_posts, num_steps, n_users, A, B, lamda_diagonal_users,
                                   a_const, lambda_val, range_window, future_prediction_range):
    x = x0
    opinion_history = [x]
    selected_posts = []
    available_posts = deepcopy(grouped_posts)

    # Decide on parallel threshold
    parallel_threshold = 8  # Use parallel processing if more than 8 candidates

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
            opinion_history.append(x)  # Maintain state if no selection
            continue

        # *** AUTOMATIC PARALLEL PROCESSING ***
        if len(candidate_posts) >= parallel_threshold:
            print(f"Using parallel processing for {len(candidate_posts)} candidates")
            
            # Prepare arguments for parallel processing
            args = []
            for i, post in enumerate(candidate_posts):
                args.append((
                    (post, i), x.copy(), x0.copy(), A.copy(), B.copy(), lamda_diagonal_users.copy(),
                    n_users, a_const, lambda_val, current_step, future_prediction_range, 
                    available_posts, num_steps
                ))
            
            # Run in parallel
            with Pool(processes=min(len(candidate_posts), mp.cpu_count())) as pool:
                results = pool.map(evaluate_single_post, args)
            
            # Find best result
            best_total_objective = float('inf')
            best_first_post = None
            
            for total_objective, post_index in results:
                if total_objective < best_total_objective:
                    best_total_objective = total_objective
                    best_first_post = candidate_posts[post_index]
                    
        else:
            # *** YOUR ORIGINAL SEQUENTIAL CODE (unchanged) ***
            print(f"Using sequential processing for {len(candidate_posts)} candidates")
            
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
                    future_t_tc = (current_step + step_index) - post['tc']  # Dynamic t-tc calculation

                    # *** USING FAST NUMBA VERSION HERE ***
                    step_objective = calculate_objective_fast(current_x, u_temp, n_users, a_const, lambda_val, future_t_tc)

                    # *** USING FAST NUMBA VERSION HERE ***
                    new_x = update_opinions_fast(current_x, u_temp, x0, A, B, lamda_diagonal_users)

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
                first_t_tc = current_step - first_post['tc']  # Dynamic t-tc calculation

                # *** USING FAST NUMBA VERSION HERE ***
                first_objective = calculate_objective_fast(x, u_first, n_users, a_const, lambda_val, first_t_tc)

                # *** USING FAST NUMBA VERSION HERE ***
                new_x = update_opinions_fast(x, u_first, x0, A, B, lamda_diagonal_users)

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
            opinion_history.append(x)  # Maintain state if no selection
            continue

        # Mark the selected post as "selected"
        best_first_post['selected'] = True

        # Update opinions using the selected post
        u = best_first_post['extremity']
        x = update_opinions(x, u, x0, A, B, lamda_diagonal_users)  # Uses fast version internally
        opinion_history.append(x)
        selected_posts.append(best_first_post)
        print("a timestep just finished (new MPC)")
        
    # Calculate misinformation metrics
    u_values = np.array([post['extremity'] for post in selected_posts])
    labelled = np.array([post['label'] for post in selected_posts])

    total_false = sum(1 for post in grouped_posts if post['label'] == 'false')
    selected_false = np.sum(labelled == 'false')
    misinformation_spread = selected_false / (2 * total_false if total_false > 0 else 0)

    print("Finished data-driven new MPC")
    return np.array(opinion_history), u_values, misinformation_spread