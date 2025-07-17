import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cvxpy as cp


def update_opinions(x, u, x0, A, B, lamda_diagonal_users):
    """Update opinions based on system dynamics"""
    # B is (n_users, 1) and u is scalar, so we need to reshape
    new_opinions = A @ x + B * u + lamda_diagonal_users @ x0
    return new_opinions


def compute_convergence_values_mb(n_users, A, B, lamda_diagonal_users, initial_user_opinion):
    """Compute theoretical Model Based steady state values"""
    n = n_users
    identity = np.eye(n)

    # Compute v vector
    v = np.linalg.inv(identity - A) @ B - np.ones((n, 1))
    vT_1n = v.T @ np.ones((n, 1))

    # Compute MB steady state
    term_1 = identity - A - (1 / vT_1n.item()) * (B @ v.T)
    term_1_inv = np.linalg.inv(term_1)
    x_star_mb = term_1_inv @ lamda_diagonal_users @ initial_user_opinion

    return x_star_mb


def theor_simulate_dynamics_mb(x0, num_steps, horizon, n_users, A, B, lamda_diagonal_users):
    """Simulate Model Based (MPC) dynamics"""
    num_steps = int(num_steps)
    horizon = int(horizon)
    n_users = int(n_users)
    x = x0
    opinion_history = [x]
    u_history = []

    # Get the target steady state
    x_mb_star = compute_convergence_values_mb(n_users, A, B, lamda_diagonal_users, x0)

    for i in range(num_steps):
        # Set up optimization variables for this time step
        x_var = cp.Variable((n_users, horizon + 1))
        u_var = cp.Variable(horizon)

        # Define objective (minimize squared distance)
        objective = cp.Minimize(
            cp.sum([cp.sum_squares(x_var[:, k] - u_var[k])
                    for k in range(horizon)])
            #+ cp.sum_squares(x_var[:, -1] - x_mb_star.flatten())
        )

        # Define constraints
        constraints = [x_var[:, 0] == x.flatten()]

        for k in range(horizon):
            # B is (n_users, 1) and u_var[k] is scalar
            constraints.append(x_var[:, k + 1] == A @ x_var[:, k] + B.flatten() * u_var[k] +
                               lamda_diagonal_users @ x0.flatten())
            constraints.append(u_var[k] >= 0)
            constraints.append(u_var[k] <= 1)

            # Terminal constraint at the end
            #if k == horizon - 1:
                #constraints.append(x_var[:, -1] == x_mb_star.flatten())

        # Solve the optimization
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        # Store the first control input
        u = u_var.value[0]
        u_history.append(u)
        #print("stored")
        #print(u)
        # Update state and store
        if i != num_steps - 1:  # Don't update on last step
            x = update_opinions(x, u, x0, A, B, lamda_diagonal_users)
            opinion_history.append(x)
    print(f"Theoretical MPC old Solved!")
    return np.array(opinion_history), np.array(u_history)