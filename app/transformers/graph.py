import os
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pickle

from app.transformers.transformer_base import Transformer


class Graph(Transformer):
    def __init__(self, min_num_common_items, max_degree):
        Transformer.__init__(self)

        self.min_num_common_items = min_num_common_items
        self.max_degree = max_degree

        self.w_mat = None
        self.b_mat = None
        self.adj_mat = None

        self.n_user = None

    def fit(self, rat_mat, do_pruning=True):
        self.n_user = rat_mat.shape[0]

        # Calc. number of common items for all pairs of the users
        mask_one = (~np.isnan(rat_mat))*1
        u_u_num_common_items = mask_one.dot(mask_one.T)

        # Init.
        self.adj_mat = np.zeros((self.n_user, self.n_user))
        self.w_mat = np.zeros((self.n_user, self.n_user))
        self.b_mat = np.zeros((self.n_user, self.n_user))
        score = np.zeros((self.n_user, self.n_user))

        # Regress all valid pairs with minimum number of common items or more
        for u in tqdm(range(self.n_user)):
            self.adj_mat[u], self.w_mat[u], self.b_mat[u], score[u] = \
                self.fit_for_user(u, u_u_num_common_items, rat_mat, self.min_num_common_items)

        # Prune the graph
        if do_pruning:
            mask_prune = Graph.prune_graph(score, self.max_degree)

            self.w_mat[~mask_prune] = 0
            self.b_mat[~mask_prune] = 0
            self.adj_mat[~mask_prune] = 0

    @staticmethod
    def fit_for_user(u, u_u_num_common_items, rat_mat, min_num_common_items):
        n_user = rat_mat.shape[0]

        adj_mat_u = np.zeros((n_user,))
        w_mat_u = np.zeros((n_user,))
        b_mat_u = np.zeros((n_user,))
        score_u = np.zeros((n_user,))

        for other_u in range(n_user):
            if u == other_u:
                continue

            if u_u_num_common_items[u, other_u] < min_num_common_items:
                continue

            x_with_nan = rat_mat[other_u]
            y_with_nan = rat_mat[u]

            # Select common items
            mask_xy = (~np.isnan(x_with_nan)) & (~np.isnan(y_with_nan))

            x = x_with_nan[mask_xy]
            y = y_with_nan[mask_xy]

            # Regress y on x
            reg = LinearRegression()
            reg.fit(X=x.reshape((-1, 1)), y=y)

            # Extract coefficients
            adj_mat_u[other_u] = 1
            w_mat_u[other_u] = reg.coef_[0]
            b_mat_u[other_u] = reg.intercept_
            score_u[other_u] = reg.score(X=x.reshape((-1, 1)), y=y) + 1e-6

        # Check if at least one edge is connected, otherwise reduce the number of required common items
        if np.sum(adj_mat_u) == 0:
            if min_num_common_items == 1:
                warnings.warn('No edge is connected to user %d!' % u)
                return adj_mat_u, w_mat_u, b_mat_u, score_u

            warn_msg = 'No user is connected to user %d with %d common items, relaxing the condition' \
                       % (u, min_num_common_items)
            warnings.warn(warn_msg)
            return Graph.fit_for_user(u, u_u_num_common_items, rat_mat, min_num_common_items - 1)

        return adj_mat_u, w_mat_u, b_mat_u, score_u

    def transform(self, data_te, **kwargs):
        return self.w_mat, self.b_mat, self.adj_mat

    def save_to_file(self, save_path, file_name, ext_dic=None):
        dic = {
            'min_num_common_items': self.min_num_common_items,
            'max_degree': self.max_degree,
            'w_mat': self.w_mat,
            'b_mat': self.b_mat,
            'adj_mat': self.adj_mat,
            'ext': ext_dic
        }

        with open(os.path.join(save_path, file_name + '.graph'), 'wb') as f:
            pickle.dump(dic, f)

    @staticmethod
    def load_from_file(load_path, file_name):
        with open(os.path.join(load_path, file_name + '.graph'), 'rb') as f:
            dic = pickle.load(f)

        g = Graph(min_num_common_items=dic['min_num_common_items'], max_degree=dic['max_degree'])
        g.w_mat = dic['w_mat']
        g.b_mat = dic['b_mat']
        g.adj_mat = dic['adj_mat']
        g.n_user = g.adj_mat.shape[0]

        return g, dic

    @staticmethod
    def prune_graph(score, max_degree):
        n_user = score.shape[0]

        mask = np.zeros((n_user, n_user)).astype(bool)

        for u in range(n_user):
            if u == 153:
                a = 1
            scores_u = score[u]

            idx_max_scores_u = np.argsort(scores_u)[-max_degree:]

            max_scores_u = scores_u[idx_max_scores_u]

            # Remove zero scores from max scores
            idx_nonzero_max_scores_u = idx_max_scores_u[max_scores_u > 0]

            mask[u, idx_nonzero_max_scores_u] = True

        return mask


if __name__ == '__main__':
    rat_mat_all = np.array([[1,     2,      3,      4],
                            [2,     4,      6,      8],
                            [2,     -1,     -4,     -7],
                            [1,     3,      3,      1],
                            [3,     7,      7,      3],
                            [-4,    4,      4,      -4]]).astype(float)

    mask_obs = np.array([[True,     True,       True,       False],
                         [True,     True,       True,       False],
                         [True,     True,       True,       False],
                         [True,     True,       True,       False],
                         [True,     True,       True,       False],
                         [True,     True,       True,       False]])

    rat_mat_obs = rat_mat_all.copy()
    rat_mat_obs[~mask_obs] = np.nan

    graph = Graph(min_num_common_items=3, max_degree=2)
    graph.fit_transform(rat_mat_obs)
