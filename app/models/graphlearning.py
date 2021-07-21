import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from app.models.graph import Graph
from app.utils.mathtools import ls


class GraphLearner:
    def __init__(self, x_mat, adj_mat, rat_mat, l2_lambda=0):
        self.x_mat = x_mat  # [2*n_user x n_item]
        self.adj_mat = adj_mat
        self.rat_mat = rat_mat
        self.l2_lambda = l2_lambda

        self.s_mat = None

    def fit(self):
        n_user = self.rat_mat.shape[0]

        self.s_mat = np.zeros((2*n_user, n_user + 1))
        self.s_mat[n_user:, -1] = np.ones((n_user,))

        for u in tqdm(range(n_user)):
            # Extract sub-matrices
            users_u = np.where(self.adj_mat[u] == 1)[0]
            idx_s_u = np.concatenate((users_u, np.array([n_user])))

            items_u = np.where(~np.isnan(self.rat_mat[u]))[0]

            x_mat_u = self.x_mat[idx_s_u][:, items_u]

            x_u = self.x_mat[u, items_u]

            # Least square estimation
            s_u = ls(x_mat_u.T, x_u, l2_lambda=self.l2_lambda)

            # Fill the s_mat
            self.s_mat[u, idx_s_u] = s_u[:, 0]


if __name__ == '__main__':
    rng = default_rng(0)

    # Simulate ratings
    n_item = 15
    min_val = 1
    max_val = 5

    r1 = rng.integers(low=min_val, high=max_val, size=(1, n_item)).astype(float)
    rat_mat_1 = np.concatenate((r1, 2*r1 + 1, -3*r1 + 5), axis=0)

    r2 = rng.integers(low=min_val, high=max_val, size=(1, n_item)).astype(float)
    rat_mat_2 = np.concatenate((r2, 3*r2 - 6, r2 + 2), axis=0)

    rat_mat_all = np.concatenate((rat_mat_1, rat_mat_2), axis=0)

    # Remove random elements
    p_miss = 0.1
    mask_nan = rng.random(size=rat_mat_all.shape) < p_miss

    rat_mat_obs = rat_mat_all.copy()
    rat_mat_obs[mask_nan] = np.nan

    # Find the graph structure
    graph = Graph.from_rating_mat(rat_mat_obs, min_num_common_items=3, max_degree=2)

    # Learn the graph shift operator
    x_mat_filled_zero = np.concatenate((rat_mat_obs, np.ones(rat_mat_obs.shape)), axis=0)
    x_mat_filled_zero[np.isnan(x_mat_filled_zero)] = 0

    g_learner = GraphLearner(x_mat_filled_zero, graph.adj, rat_mat_obs, l2_lambda=0.1)

    g_learner.fit()

