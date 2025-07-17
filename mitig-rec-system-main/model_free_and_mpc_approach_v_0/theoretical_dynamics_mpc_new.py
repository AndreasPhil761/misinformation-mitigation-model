import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cvxpy as cp


def update_opinions(x, u, x0, A, B, lamda_diagonal_users):
    """Update opinions based on system dynamics"""
    new_opinions = A @ x + B * u + lamda_diagonal_users @ x0
    return new_opinions


def compute_convergence_theoretical_dynamics_mpc_new(n_users, A, B, Lambda, x0, a_const, lambda_val):
    """Compute theoretical Model Based steady state values"""
    I = np.eye(n_users)
    inv_I_minus_A = np.linalg.inv(I - A)

    # Calculate v = (I - A)^-1 * B
    v = inv_I_minus_A @ B

    # Calculate z = (I - A)^-1 * Lambda * x0
    z = inv_I_minus_A @ (Lambda @ x0)

    # Calculate parameters needed for u*_MB
    ones = np.ones((n_users, 1))
    q = -ones.T @ v + n_users + v.T @ v - v.T @ ones
    q = float(q)  # Convert from 1x1 matrix to scalar

    s = -(v.T @ z - ones.T @ z)
    s = float(s)  # Convert from 1x1 matrix to scalar

    # Calculate u*_MB (without time-dependent term, as in the steady-state case)
    u_star_MB = s / (q + a_const*np.exp(-lambda_val))

    # Ensure u is within bounds [0, 1]
    #u_star_MB = max(0, min(1, u_star_MB))
    #print("q")
    #print(q)
    #print("s")
    #print(s)
    # Calculate x*_MB
    x_star_MB = v * u_star_MB + z
    #print("u_mb")
    #print(u_star_MB)
    return x_star_MB


def theor_simulate_dynamics_mb_new(x0, num_steps, horizon, n_users, A, B, lamda_diagonal_users, a_const, lambda_val):
    """Simulate Model Based (MPC) dynamics with new theta function"""
    num_steps = int(num_steps)
    horizon = int(horizon)
    n_users = int(n_users)
    x = x0
    opinion_history = [x]
    u_history = []

    # Get the target steady state
    x_mb_star = compute_convergence_theoretical_dynamics_mpc_new(n_users, A, B, lamda_diagonal_users, x0, a_const, lambda_val)

    for i in range(num_steps):
        # Set up optimization variables for this time step
        x_var = cp.Variable((n_users, horizon + 1))
        u_var = cp.Variable(horizon)

        # Define objective with new theta function
        stage_cost = 0
        for k in range(horizon):
            # Original theta term
            original_theta = cp.sum_squares(x_var[:, k] - u_var[k])

            # EN term (novelty + extremity)
            en_term = a_const * cp.square(u_var[k]) * cp.exp(-lambda_val)

            # Combined cost for this stage
            stage_cost += original_theta + en_term

        # Terminal cost
        #terminal_cost = cp.sum_squares(x_var[:, -1] - x_mb_star.flatten())

        # Total objective
        objective = cp.Minimize(stage_cost) #+ #terminal_cost)

        # Define constraints
        constraints = [x_var[:, 0] == x.flatten()]

        for k in range(horizon):
            # B is (n_users, 1) and u_var[k] is scalar
            constraints.append(x_var[:, k + 1] == A @ x_var[:, k] + B.flatten() * u_var[k] +
                               lamda_diagonal_users @ x0.flatten())
            constraints.append(u_var[k] >= 0)
            constraints.append(u_var[k] <= 1)
        
        
        #tolerance = 0.02
        #constraints.append(x_var[:, -1] >= x_mb_star.flatten() - tolerance)
        #constraints.append(x_var[:, -1] <= x_mb_star.flatten() + tolerance)

        # Solve the optimization
        problem = cp.Problem(objective, constraints)
        #problem.solve(solver=cp.OSQP)
        problem.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6)
        # Store the first control input
        u = u_var.value[0]
        u_history.append(u)

        # Update state and store
        if i != num_steps - 1:  # Don't update on last step
            x = update_opinions(x, u, x0, A, B, lamda_diagonal_users)
            opinion_history.append(x)
    print(f"Theoretical MPC new Solved!")
    return np.array(opinion_history), np.array(u_history)