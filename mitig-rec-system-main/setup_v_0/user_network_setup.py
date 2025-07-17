#-------------------------dependencies----------------------------
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from radical_user_network import create_radical_user
from radical_user_network import create_radical_user_paper_copy
from adjacency_graph import create_graph

#-------------------------constants-------------------------------
n_users = 100 # only users, recommender exists by default
h_stub, l_stub = 0, 0.05 # highest and lowest stubbornness in the network
seed_value = 48 # seed value to keep randomness "constant" until change
np.random.seed(seed_value)
interconnectivity_users = 0.25
interconnectivity_rec = 0.8
t=0 #special case of radical user
tp=0 #special case of radical user paper copy
make_graph=0 #print simple adjacency graph

#-------------------------interconnectivity-----------------------
total_possible_user_connections = n_users**2  # user-to-user (including self-connections)
total_possible_rec_connections = n_users      # only rec-to-users connections
total_meaningful_connections = total_possible_user_connections + total_possible_rec_connections

num_user_connections = int(total_possible_user_connections * interconnectivity_users)
num_rec_connections = int(total_possible_rec_connections * interconnectivity_rec)

mask = np.zeros((n_users+1, n_users+1))

# 1. User-to-user connections
user_indices = [(i, j) for i in range(n_users) for j in range(n_users)]
np.random.seed(seed_value)
selected_user_indices = np.random.choice(len(user_indices), num_user_connections, replace=False)
for idx in selected_user_indices:
    i, j = user_indices[idx]
    mask[i, j] = 1

# 2. Recommender-to-user connections (rec affects random users)
rec_indices = [(i, n_users) for i in range(n_users)]  # FIXED: varying row, fixed column
selected_rec_indices = np.random.choice(len(rec_indices), num_rec_connections, replace=False)
for idx in selected_rec_indices:
    i, j = rec_indices[idx]
    mask[i, j] = 1

#-------------------------user stubbornness-----------------------
usr_lamda = h_stub + (l_stub - h_stub) * np.random.rand(n_users+1)
lamda_diagonal = np.diag(usr_lamda)

#-------------------------initial user opinions-------------------
#n_trials = 50
#p_success = 0.5
#binomial_samples = np.random.binomial(n=n_trials, p=p_success, size=n_users+1)
#in_usr_op = (binomial_samples / n_trials).reshape(-1, 1)
# Parameters for the Beta distribution (adjust alpha and beta)
alpha = 7   # Higher alpha biases toward 1
beta = 2    # Higher beta reduces bias toward 0

# Generate biased random numbers
in_usr_op = np.random.beta(alpha, beta, size=(n_users + 1, 1))
#in_usr_op = np.random.uniform(0, 1, size=(n_users+1, 1))

#-------------------------graph relation calculation--------------
W = np.random.rand(n_users+1, n_users+1)
W = W * mask
prejudiced_users = [i for i, lamda in enumerate(usr_lamda) if float(lamda) > 0.0]
for i in range(n_users+1):
    if i not in prejudiced_users:
        # Connect to a random prejudiced user
        target = np.random.choice(prejudiced_users)
        W[i, target] = np.random.rand()
#np.fill_diagonal(W, 0)
for i in range(n_users):
    if np.sum(W[i, :]) == 0 or (np.sum(W[i, :]) == W[i, i] and W[i, i] != 0):
        while True:
            j = np.random.choice(range(n_users+1))
            if j != i:
                W[i, j] = np.random.rand()
                break
if np.sum(W[:,n_users]) == 0:
    j = np.random.choice(range(n_users))
    W[j,n_users] = np.random.rand()
row_sums = W.sum(axis=1)
row_sums[row_sums == 0] = 1
W = W / row_sums[:, np.newaxis]

#-------------------------separating users from recommender------
W_users = W[:n_users, :n_users]
W_rec = W[:-1, -1].reshape(-1, 1)
lamda_diagonal_users = lamda_diagonal[:-1, :-1]
initial_user_opinion = in_usr_op[:-1]

#-------------------------radical user case----------------------
if t == 1:
    initial_user_opinion, user_lamda, lamda_diagonal_users, W_users, W_rec = create_radical_user(t, n_users, in_usr_op, usr_lamda, W, lamda_diagonal)

#-------------------------radical user paper copy case-----------
if tp == 1:
    initial_user_opinion, user_lamda, lamda_diagonal_users, W_users, W_rec = create_radical_user_paper_copy()

#-------------------------LTI formulation------------------------
A = (np.eye(n_users) - lamda_diagonal_users) @ W_users
B = (np.eye(n_users) - lamda_diagonal_users) @ W_rec
I_n = np.eye(n_users)
inverse_term = np.linalg.inv(I_n - A)
l = inverse_term @ (lamda_diagonal_users @ initial_user_opinion)
m = inverse_term @ (B + lamda_diagonal_users @ initial_user_opinion)

#-------------------------printing for confirmation--------------
print("W matrix:")
print(W)
print("W matrix users:")
print(W_users)
print("w recommender:")
print(W_rec)
print("initial user opinions:")
print(initial_user_opinion)
print("lamda diagonal of users")
print(lamda_diagonal_users)
print("opinion min")
print(l)
print("opinion max")
print(m)

#-------------------------export dependencies--------------------
np.savez('initial_dependencies.npz',
         initial_user_opinion=initial_user_opinion,
         n_users=n_users,
         lamda_diagonal_users=lamda_diagonal_users,
         W_users=W_users,
         W_rec=W_rec,
         A=A,
         B=B,
         l=l,
         m=m)

#-------------------------print graph directly-------------------
if make_graph == 1:
    create_graph(n_users,W_users,W_rec)