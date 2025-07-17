import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cvxpy as cp

def update_opinions(x, u, x0, A, B, lamda_diagonal_users):
    new_opinions = A @ x + B * u + lamda_diagonal_users @ x0
    return new_opinions

def theor_simulate_dynamics_mf(x0, u, num_steps, n_users, A, B, lamda_diagonal_users):
    x = x0
    opinion_history = [x]
    u_history = []
    for i in range(num_steps):
        #u = np.mean(x)
        #u_temp = cp.Variable()
        sum_x_squared = np.sum(x ** 2)
        sum_x = np.sum(x)
        sum_ones = n_users
        u_temp = cp.Variable()
        objective = cp.Minimize(sum_x_squared - 2 * u_temp * sum_x + u_temp ** 2 * sum_ones)
        #objective = cp.Minimize(cp.norm(x - u_temp * np.ones(n_users),2) ** 2)
        constraints = []
        constraints.append(u_temp >= 0)
        constraints.append(u_temp <= 1)
        #print(f"Solving...")
        problem = cp.Problem(objective, constraints)
        #problem.solve(solver=cp.ECOS, max_iters=10000, eps=1e-8, alpha=1.5, use_indirect=False)
        problem.solve(solver=cp.OSQP)
        #print(f"Solved!")
        u = u_temp.value
        if i != 0:
            x = update_opinions(x, u, x0, A, B, lamda_diagonal_users)
            opinion_history.append(x)
        u_history.append(u_temp.value)
    print(f"Theoretical MF old Solved!")
    return np.array(opinion_history), np.array(u_history)

def compute_convergence_values(n_users, A, B, lamda_diagonal, initial_user_opinion):
    n = n_users
    identity = np.eye(n)
    s_mf = np.linalg.inv(identity - A - (1 / n) * B @ np.ones((1, n))) @ lamda_diagonal
    x_star_mf = s_mf @ initial_user_opinion
    return x_star_mf