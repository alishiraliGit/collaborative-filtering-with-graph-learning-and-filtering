import numpy as np
from sklearn.linear_model import LinearRegression


class Graph:
    def __init__(self, s, adj, min_num_common_items, max_degree):
        self.min_num_common_items = min_num_common_items
        self.max_degree = max_degree

        self.s = s
        self.adj = adj

    @staticmethod
    def from_rating_mat(rat_mat, min_num_common_items, max_degree):
        n_user, _ = rat_mat.shape

        # Calc. number of common items for all pairs of the users
        mask_one = ~np.isnan(rat_mat)*1
        u_u_num_common_items = mask_one.dot(mask_one.T)

        # Init.
        w = np.zeros((n_user, n_user))
        b = np.zeros((n_user, n_user))
        score = np.zeros((n_user, n_user))

        # Regress all valid pairs with minimum number of common items or more
        for u1 in range(n_user):
            for u2 in range(n_user):
                if u1 == u2:
                    continue

                if u_u_num_common_items[u1, u2] < min_num_common_items:
                    continue

                x_with_nan = rat_mat[u1]
                y_with_nan = rat_mat[u2]

                # Select common items
                mask_xy = (~np.isnan(x_with_nan)) & (~np.isnan(y_with_nan))

                x = x_with_nan[mask_xy]
                y = y_with_nan[mask_xy]

                # Regress y on x
                reg = LinearRegression()
                reg.fit(X=x.reshape((-1, 1)), y=y)

                # Extract coefficients
                w[u2, u1] = reg.coef_[0]
                b[u2, u1] = reg.intercept_
                score[u2, u1] = reg.score(X=x.reshape((-1, 1)), y=y)

        # Prune the graph
        mask_prune = Graph.prune_graph(score, max_degree)

        w_pruned = np.zeros((n_user, n_user))
        b_pruned = np.zeros((n_user, n_user))

        w_pruned[mask_prune] = w[mask_prune]
        b_pruned[mask_prune] = b[mask_prune]

        # Calc. the shift operator
        s = np.concatenate((w_pruned, b_pruned), axis=1)

        adj = mask_prune*1

        return Graph(s, adj, min_num_common_items, max_degree)

    @staticmethod
    def prune_graph(score, max_degree):
        n_user = score.shape[0]

        mask = np.zeros((n_user, n_user)).astype(bool)

        for u in range(n_user):
            scores_u = score[u]

            idx_max_scores_u = np.argsort(scores_u)[-max_degree:]

            max_scores_u = scores_u[idx_max_scores_u]

            # Remove zero scores from max scores
            idx_nonzero_max_scores_u = idx_max_scores_u[max_scores_u > 0]

            # Check that at least one edge is remained
            assert len(idx_nonzero_max_scores_u) > 0, 'no input edge to user %d!' % u

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

    graph = Graph.from_rating_mat(rat_mat_obs, 3, 2)
