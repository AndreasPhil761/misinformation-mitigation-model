import numpy as np

def create_radical_user(t, n_users, in_usr_op, user_lamda, W, lamda_diagonal):
    radical_user = np.random.choice(range(n_users))

    print(f"user {radical_user + 1} is a radical")

    in_usr_op[radical_user] = np.random.choice([0, 1])

    user_lamda[radical_user] = 1
    lamda_diagonal = np.diag(user_lamda)

    W[:, radical_user] = 0

    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W = W / row_sums[:, np.newaxis]

    W_users = W[:n_users, :n_users]
    W_rec = W[:-1, -1].reshape(-1, 1)
    lamda_diagonal_users = lamda_diagonal[:-1, :-1]
    initial_user_opinion = in_usr_op[:-1]

    return initial_user_opinion, user_lamda, lamda_diagonal_users, W_users, W_rec

def create_radical_user_paper_copy():
    n_users = 6
    initial_user_opinion = np.array([0.67, 0.74, 0.83, 0.68, 0, 0.59]).reshape(-1, 1)
    user_lamda = np.array([0.011, 0.001, 0.092, 0.064, 1.000, 0.055])
    lamda_diagonal_users = np.diag(user_lamda)
    W = np.zeros((n_users + 1, n_users + 1))

    W[0, 0] = 0.0
    W[0, 1] = 0.041
    W[0, 2] = 0.0
    W[0, 3] = 0.397
    W[0, 4] = 0.0
    W[0, 5] = 0.0
    W[0, 6] = 0.562
    W[1, 0] = 0.0
    W[1, 1] = 0.191
    W[1, 2] = 0.0
    W[1, 3] = 0.0
    W[1, 4] = 0.0
    W[1, 5] = 0.011
    W[1, 6] = 0.798
    W[2, 0] = 0.0
    W[2, 1] = 0.0
    W[2, 2] = 0.0
    W[2, 3] = 0.0
    W[2, 4] = 0.0
    W[2, 5] = 0.224
    W[2, 6] = 0.776
    W[3, 0] = 1
    W[3, 1] = 0.0
    W[3, 2] = 0.0
    W[3, 3] = 0.0
    W[3, 4] = 0.0
    W[3, 5] = 0.0
    W[3, 6] = 0.0
    W[4, 0] = 0
    W[4, 1] = 0.0
    W[4, 2] = 0.472
    W[4, 3] = 0.357
    W[4, 4] = 0.0
    W[4, 5] = 0.171
    W[4, 6] = 0.0
    W[5, 0] = 0.0
    W[5, 1] = 0.0
    W[5, 2] = 0.0
    W[5, 3] = 1
    W[5, 4] = 0.0
    W[5, 5] = 0.0
    W[5, 6] = 0
    W[6, 0] = 0
    W[6, 1] = 0
    W[6, 2] = 0
    W[6, 3] = 0.0
    W[6, 4] = 0.0
    W[6, 5] = 0.0
    W[6, 6] = 0.0

    W_users = W[:n_users, :n_users]
    W_rec = W[:-1, -1].reshape(-1, 1)

    return initial_user_opinion, user_lamda, lamda_diagonal_users, W_users, W_rec