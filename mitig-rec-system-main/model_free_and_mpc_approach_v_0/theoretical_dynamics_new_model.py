import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cvxpy as cp

def update_opinions(x, u, x0, A, B, lamda_diagonal_users):
    new_opinions = A @ x + B * u + lamda_diagonal_users @ x0
    return new_opinions


def compute_convergence_theoretical_dynamics_mf(n_users, A, B, Lambda, x0, a_const, lambda_val):
    """Compute theoretical Model Free steady state values"""
    # Create identity matrix
    I = np.eye(n_users)

    # Ensure B is a column vector

    # Create a row vector of ones
    ones_row = np.ones((1, n_users))

    # Calculate the denominator term: n(1 + a_const * exp(-lambda_val))
    denominator = n_users + a_const * np.exp(-lambda_val)

    # Calculate the term B * 1_n^T / denominator
    B_ones_term = (1 / denominator) * B @ ones_row

    # Calculate the matrix to invert: I_n - A - B * 1_n^T / denominator
    matrix_to_invert = I - A - B_ones_term

    # Invert the matrix
    inverted_matrix = np.linalg.inv(matrix_to_invert)

    # Calculate x*_MF = inverted_matrix * Lambda * x0
    x_star_MF = inverted_matrix @ (Lambda @ x0)

    # Calculate u*_MF (the average of x*_MF)
    #u_star_MF = np.mean(x_star_MF)

    #print("u_mf")
    #print(u_star_MF)

    return x_star_MF

def theor_simulate_dynamics_mf_new(x0, u, num_steps, n_users, A, B, lamda_diagonal_users, a_const, lambda_val):
    x = x0
    opinion_history = [x]
    u_history = []
    for i in range(num_steps):

        sum_x_squared = np.sum(x ** 2)
        sum_x = np.sum(x)
        sum_ones = n_users
        #theoretical_max = np.linalg.norm( -1 * np.ones(n_users) - np.ones(n_users), ord = 2)
        u_temp = cp.Variable()
        #-------------------------original term optimization-----------------------------
        original_theta_term = sum_x_squared - 2 * u_temp * sum_x + u_temp ** 2 * sum_ones
        #objective_original = cp.Minimize(original_theta_term)
        #problem_original = cp.Problem(objective_original)
        #problem_original.solve(solver=cp.OSQP)

        # -------------------------define ratio approximation----------------------------
        #original_theta_term = original_theta_term / theoretical_max
        #k = a_const * theta_magnitude

        #-------------------------EN term, novelty + extremity term----------------------
        en_term = a_const * cp.square(cp.abs(u_temp)) * cp.exp(-lambda_val)

        #-------------------------full obj. function-------------------------------------
        objective = cp.Minimize(original_theta_term + en_term)

        #-------------------------constraints--------------------------------------------
        constraints = []
        constraints.append(u_temp >= 0)
        constraints.append(u_temp <= 1)

        #-------------------------optimization solver------------------------------------
        #print(f"Solving...")
        problem = cp.Problem(objective, constraints)
        #problem.solve(solver=cp.ECOS, max_iters=10000, eps=1e-8, alpha=1.5, use_indirect=False)
        problem.solve(solver=cp.OSQP)
        #print(f"Solved!")
        #-------------------------data saving--------------------------------------------
        u = u_temp.value
        if i != 0:
            x = update_opinions(x, u, x0, A, B, lamda_diagonal_users)
            opinion_history.append(x)
        u_history.append(u_temp.value)
    print(f"Theoretical MF new Solved!")
    return np.array(opinion_history), np.array(u_history)