import os
import abc

from tqdm import tqdm
import pickle
import numpy as np
from scipy.optimize import least_squares, line_search, lsq_linear
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet, Lasso
from scipy.sparse.linalg import eigsh

from app.utils.log import print_red
from app.models.model_base import Model
from app.transformers.graph import Graph, SymmetricGraph
from app.utils.mathtools import ls, list_minus_list, vectorize, unvectorize


class GraphLearnerBase(Model, abc.ABC):
    def __init__(self, adj_mat, ui_is_rated_mat):
        self._adj_mat = adj_mat  # [n_user x n_user]
        self._ui_is_rated_mat = ui_is_rated_mat  # [n_user x n_item]
        self._n_user, self._n_item = ui_is_rated_mat.shape

        self.x_mat = None  # [(n_user + (1)) x n_item]
        self.s_mat = None  # [(n_user + (1)) x (n_user + (1))]

    @staticmethod
    def from_graph_object(g, ui_is_rated_mat):
        pass

    @abc.abstractmethod
    def fit_shift_operator(self, **kwargs):
        pass

    @abc.abstractmethod
    def fit_x(self, **kwargs):
        pass

    def fit(self, x_0_mat, verbose=False, **kwargs):
        self.x_mat = x_0_mat

        if verbose:
            print('Updating x_mat ...')
        self.fit_x(**kwargs)

        if verbose:
            print('Updating s_mat ...')
        self.fit_shift_operator(**kwargs)

    def fit_scale_of_shift_operator(self, weights=np.array([1]), **_kwargs):
        # Predict rating with current shift operator
        y_pr_mat = self.predict(self.x_mat)

        # Find rated users and items
        users_rated, items_rated = np.where(self._ui_is_rated_mat)

        # Assign weights to ratings based on their items
        if len(weights) > 1:
            weights_ratings = weights[items_rated]
        else:
            weights_ratings = weights[0]

        # Find the best scale
        reg = LinearRegression(fit_intercept=False)

        reg.fit(
            X=y_pr_mat[self._ui_is_rated_mat].reshape((-1, 1)),
            y=self.x_mat[:self._n_user][self._ui_is_rated_mat],
            sample_weight=weights_ratings
        )

        scale = reg.coef_[0]

        # Scale the shift operator
        self.s_mat[:self._n_user] *= scale


class GraphLearner(GraphLearnerBase):

    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

        self._zero_one_design_mat, self._users_x, self._items_x, self._uu_to_idx_dic =\
            self.fit_design_matrix(adj_mat, ui_is_rated_mat)

    @staticmethod
    def from_graph_object_asymmetric(g: Graph, ui_is_rated_mat):
        # Instantiate a GraphLearner
        adj_mat = g.adj_mat
        g_learner = GraphLearner(adj_mat, ui_is_rated_mat)

        # Init. s_mat from graph w & b
        n_user = g.n_user
        s_mat = np.zeros((n_user + 1, n_user + 1))
        s_mat[n_user, n_user] = 1

        n_in_edge = np.sum(adj_mat, axis=1)

        n_in_edge[n_in_edge == 0] = 1

        b_normalized = np.sum(g.b_mat, axis=1) / n_in_edge
        w_normalized_mat = np.diag(1 / n_in_edge).dot(g.w_mat)

        s_mat[:n_user, :n_user] = w_normalized_mat
        s_mat[:n_user, n_user] = b_normalized

        g_learner.s_mat = s_mat

        return g_learner

    @staticmethod
    def from_graph_object(g: SymmetricGraph, ui_is_rated_mat):
        n_user = g.n_user
        adj_mat = g.adj_mat
        w_mat = g.w_mat

        # Instantiate a GraphLearner
        g_learner = GraphLearner(adj_mat, ui_is_rated_mat)

        # Find largest eigenvalue
        w_1 = eigsh(w_mat, k=1, which='LM', return_eigenvectors=False)[0]

        # Set shift operator as the normalized the weight matrix
        s_mat = np.zeros((n_user + 1, n_user + 1))
        s_mat[n_user, n_user] = 1
        s_mat[:, n_user] = 1  # ToDo
        s_mat[:n_user, :n_user] = w_mat/w_1

        g_learner.s_mat = s_mat

        return g_learner

    @staticmethod
    def fit_design_matrix(adj_mat, ui_is_rated_mat):
        n_user = adj_mat.shape[0]

        # Assign an index to each connected user-user pair
        idx = 0
        uu_to_idx_dic = {}
        for u_1, u_2 in np.argwhere(adj_mat == 1):
            if u_1 > u_2:
                uu_to_idx_dic[tuple([u_1, u_2])] = idx
                idx += 1

        for u in range(n_user):
            uu_to_idx_dic[(n_user, u)] = idx
            idx += 1

        # Init.
        users_x = []
        items_x = []
        zero_one_design_mat = np.zeros((np.sum(ui_is_rated_mat), len(uu_to_idx_dic)))

        # For on all rated user-item pairs
        for idx_rating, ui in tqdm(enumerate(np.argwhere(ui_is_rated_mat)), desc='GraphLearner:fit_design_matrix'):
            u, i = ui

            # Find u's neighbors
            neighbors_u = np.where(adj_mat[u] == 1)[0]

            # Find uu indices corresponding to u and u's neighbors
            uu_indices = \
                [uu_to_idx_dic[(np.maximum(neighbor_u, u), np.minimum(neighbor_u, u))] for neighbor_u in neighbors_u]

            # Add a row to the zero-one design matrix
            zero_one_design_mat[idx_rating, uu_indices] = 1
            zero_one_design_mat[idx_rating, uu_to_idx_dic[(n_user, u)]] = 1

            # Keep a reference of u's neighbors and i to later fill the design matrix
            users_x.extend(neighbors_u.tolist())
            users_x.append(n_user)

            items_x.extend([i]*len(neighbors_u))
            items_x.append(i)

        return zero_one_design_mat, users_x, items_x, uu_to_idx_dic

    def fit_shift_operator_asymmetric(self, l2_lambda_s=0., weights=np.array([1]), verbose_s=True, **_kwargs):
        n_user = self._n_user

        self.s_mat = np.zeros((n_user + 1, n_user + 1))
        self.s_mat[n_user, n_user] = 1

        for u in tqdm(range(n_user), disable=not verbose_s, desc='GraphLearner:fit_shift_operator'):
            # Extract sub-matrices
            users_u = np.where(self._adj_mat[u] == 1)[0]
            idx_s_u = np.concatenate((users_u, [n_user]))

            items_u = np.where(self._ui_is_rated_mat[u])[0]

            x_mat_u = self.x_mat[idx_s_u][:, items_u]

            x_u = self.x_mat[u, items_u]

            if len(weights) > 1:
                weights_u = weights[items_u]
            else:
                weights_u = weights

            # Least square estimation
            s_u = ls(x_mat_u.T, x_u, l2_lambda=l2_lambda_s, weights=weights_u)

            # Fill the s_mat
            self.s_mat[u, idx_s_u] = s_u[:, 0]

    def fit_shift_operator(self, l2_lambda_s=0., l1_ratio_s=0, weights=np.array([1]), verbose_s=True, **_kwargs):
        # Find the design matrix
        design_mat = self._zero_one_design_mat.copy()
        design_mat[np.where(design_mat == 1)] = self.x_mat[(self._users_x, self._items_x)]

        # Find rated users and items
        users_rated, items_rated = np.where(self._ui_is_rated_mat)

        # Extract the target ratings
        y = self.x_mat[:-1][(users_rated, items_rated)]

        # Assign weights to ratings based on their items
        if len(weights) > 1:
            weights_ratings = weights[items_rated]
        else:
            weights_ratings = weights[0]

        # Solve the linear system
        if verbose_s:
            print_red('GraphLearner:fit_shift_operator:solve_linear_system')

        # s = ls(design_mat, y, l2_lambda=l2_lambda_s, weights=weights_ratings)
        if l1_ratio_s == 0:
            reg = Ridge(alpha=l2_lambda_s, fit_intercept=False)
        elif l1_ratio_s == 1:
            reg = Lasso(alpha=l2_lambda_s/len(y), fit_intercept=False)
        else:
            reg = ElasticNet(alpha=l2_lambda_s/len(y), l1_ratio=l1_ratio_s, fit_intercept=False)

        reg.fit(X=design_mat, y=y, sample_weight=weights_ratings)
        s = reg.coef_

        # Fill the shift op matrix
        for uu, idx in self._uu_to_idx_dic.items():
            u_1, u_2 = uu
            self.s_mat[u_1, u_2] = s[idx]
            self.s_mat[u_2, u_1] = s[idx]

    def fit_x(self, min_val=1, max_val=5, max_distance_to_rated=1, gamma=0., max_nfev_x=3,
              verbose_x=True, **kwargs):

        n_user, n_item = self._n_user, self._n_item

        new_x_mat = self.x_mat.copy()

        for it in tqdm(range(n_item), disable=not verbose_x, desc='GraphLearner:fit_x'):
            # Extract sub-matrices
            unrated_users_i = np.where(~self._ui_is_rated_mat[:, it])[0]
            rated_users_wo_bias_i = np.where(self._ui_is_rated_mat[:, it])[0]
            rated_users_i = np.concatenate((rated_users_wo_bias_i, [n_user]))

            # Find the unrated users which are far from rated users
            rated_users_one_hot = self._ui_is_rated_mat[:, it] * 1
            adj_d_mat = self._adj_mat + self._adj_mat.T
            n_connected_rated = adj_d_mat.dot(rated_users_one_hot.reshape((-1, 1)))

            for _ in range(1, max_distance_to_rated):
                n_connected_rated = adj_d_mat.dot(n_connected_rated)

            far_to_be_updated_users = np.where(n_connected_rated == 0)[0]

            # Consider far users as rated
            unrated_users_i = np.array(list_minus_list(unrated_users_i, far_to_be_updated_users))
            rated_users_i = np.concatenate((rated_users_i, far_to_be_updated_users))

            n_unrated_users_i = len(unrated_users_i)
            if n_unrated_users_i == 0:
                continue

            s_minus_eye_mat = self.s_mat - np.eye(n_user + 1)
            s_unrated_i_mat = s_minus_eye_mat[:, unrated_users_i]
            s_rated_i_mat = s_minus_eye_mat[:, rated_users_i]

            x_rated_i = self.x_mat[rated_users_i, it:(it + 1)]

            y_i = -s_rated_i_mat.dot(x_rated_i)

            # Add stability term
            y_i_extended = np.concatenate(
                (y_i, np.sqrt(gamma) * np.mean(self.x_mat[unrated_users_i], axis=1).reshape((-1, 1))),
                axis=0
            )
            s_unrated_i_extended_mat = \
                np.concatenate((s_unrated_i_mat, np.sqrt(gamma) * np.eye(n_unrated_users_i)), axis=0)

            # Least square estimation
            x_unrated_i = self.x_mat[unrated_users_i, it:(it + 1)]
            x_unrated_i_new = self.ls(x_0=x_unrated_i[:, 0],
                                      s_unrated_mat=s_unrated_i_extended_mat,
                                      y=y_i_extended,
                                      min_val=min_val,
                                      max_val=max_val,
                                      max_nfev=max_nfev_x)

            # Update the x_mat
            new_x_mat[unrated_users_i, it] = x_unrated_i_new[:, 0]

        self.x_mat = new_x_mat

    @staticmethod
    def ls_fun(x, s_unrated_mat, y, *_args, **_kwargs):
        f = s_unrated_mat.dot(x.reshape((-1, 1))) - y

        return f[:, 0]

    @staticmethod
    def ls_jac(_x, s_unrated_mat, *_args, **_kwargs):
        return s_unrated_mat

    @staticmethod
    def ls(x_0, s_unrated_mat, y, min_val, max_val, max_nfev,
           method='trf', loss='linear', f_scale=1, verbose=0, **kwargs):

        res = least_squares(
            fun=GraphLearner.ls_fun,
            jac=GraphLearner.ls_jac,
            x0=x_0,
            args=(s_unrated_mat, y),
            bounds=(min_val, max_val),
            method=method,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            verbose=verbose,
            **kwargs
        )

        return res.x.reshape((-1, 1))

    def predict(self, x_mat, **kwargs):
        return self.s_mat.dot(x_mat)[:-1]

    def save_to_file(self, savepath, filename, ext_dic=None):
        dic = {
            'x_mat': self.x_mat,
            's_mat': self.s_mat,
            'adj_mat': self._adj_mat,
            'ui_is_rated_mat': self._ui_is_rated_mat,
            'ext': ext_dic
        }

        with open(os.path.join(savepath, filename + '.graphlearner'), 'wb') as f:
            pickle.dump(dic, f)

    @staticmethod
    def load_from_file(loadpath, filename):
        with open(os.path.join(loadpath, filename + '.graphlearner'), 'rb') as f:
            dic = pickle.load(f)

        g_learner = GraphLearner(adj_mat=dic['adj_mat'], ui_is_rated_mat=dic['ui_is_rated_mat'])

        g_learner.x_mat = dic['x_mat']
        g_learner.s_mat = dic['s_mat']

        return g_learner, dic


class GraphLearnerWithNoBias(GraphLearnerBase):

    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

        self._zero_one_design_mat, self._users_x, self._items_x, self._uu_to_idx_dic =\
            self.fit_design_matrix(adj_mat, ui_is_rated_mat)

    @staticmethod
    def from_graph_object(g: SymmetricGraph, ui_is_rated_mat):
        adj_mat = g.adj_mat
        w_mat = g.w_mat

        # Instantiate a GraphLearner
        g_learner = GraphLearnerWithNoBias(adj_mat, ui_is_rated_mat)

        # Find the largest eigenvalue
        w_1 = eigsh(w_mat, k=1, which='LM', return_eigenvectors=False)[0]

        # Set the shift operator as the normalized weight matrix
        g_learner.s_mat = w_mat/w_1

        return g_learner

    @staticmethod
    def fit_design_matrix(adj_mat, ui_is_rated_mat):
        # Assign an index to each connected user-user pair
        idx = 0
        uu_to_idx_dic = {}
        for u_1, u_2 in np.argwhere(adj_mat == 1):
            if u_1 > u_2:
                uu_to_idx_dic[tuple([u_1, u_2])] = idx
                idx += 1

        # Init.
        users_x = []
        items_x = []
        zero_one_design_mat = np.zeros((np.sum(ui_is_rated_mat), len(uu_to_idx_dic)))

        # For on all rated user-item pairs
        for idx_rating, ui in enumerate(tqdm(np.argwhere(ui_is_rated_mat), desc='GraphLearnerWNB:fit_design_matrix')):
            u, i = ui

            # Find u's neighbors
            neighbors_u = np.where(adj_mat[u] == 1)[0]

            # Find uu indices corresponding to u and u's neighbors
            uu_indices = \
                [uu_to_idx_dic[(np.maximum(neighbor_u, u), np.minimum(neighbor_u, u))] for neighbor_u in neighbors_u]

            # Add a row to the zero-one design matrix
            zero_one_design_mat[idx_rating, uu_indices] = 1

            # Keep a reference of u's neighbors and i to later fill the design matrix
            users_x.extend(neighbors_u.tolist())

            items_x.extend([i]*len(neighbors_u))

        return zero_one_design_mat, users_x, items_x, uu_to_idx_dic

    def fit_shift_operator(self, l2_lambda_s=0., l1_ratio_s=0., max_iter_s=20, bound_s=True,
                           weights=np.array([1]), verbose_s=True, **_kwargs):
        # Find the design matrix
        design_mat = self._zero_one_design_mat.copy()
        design_mat[np.where(design_mat == 1)] = self.x_mat[(self._users_x, self._items_x)]

        # Find rated users and items
        users_rated, items_rated = np.where(self._ui_is_rated_mat)

        # Extract the target ratings
        y = self.x_mat[(users_rated, items_rated)]

        # Weight ratings
        if len(weights) > 1:
            weights_ratings = weights[items_rated]
            weights_ratings /= np.mean(weights_ratings)  # Normalize weights

            design_mat *= weights_ratings.reshape((-1, 1))
            y *= weights_ratings

        # Solve the linear system
        if verbose_s:
            print_red('GraphLearnerWNB:fit_shift_operator:solve_linear_system')

        n_rat, n_s = design_mat.shape

        if bound_s:
            if l2_lambda_s > 0:
                design_eye_mat = np.concatenate((design_mat, l2_lambda_s*np.eye(n_s)), axis=0)
                y_zero_pad = np.concatenate((y, np.zeros((n_s,))))
            else:
                design_eye_mat = design_mat
                y_zero_pad = y

            s = lsq_linear(
                A=design_eye_mat,
                b=y_zero_pad,
                bounds=(0, np.inf),
                method='trf',
                max_iter=max_iter_s,
                verbose=2*verbose_s
            ).x
        else:
            if l1_ratio_s == 0:
                reg = Ridge(alpha=l2_lambda_s, fit_intercept=False)
            elif l1_ratio_s == 1:
                reg = Lasso(alpha=l2_lambda_s/len(y), fit_intercept=False)
            else:
                reg = ElasticNet(alpha=l2_lambda_s/len(y), l1_ratio=l1_ratio_s, fit_intercept=False)

            reg.fit(X=design_mat, y=y)
            s = reg.coef_

        # Fill the shift op matrix
        for uu, idx in self._uu_to_idx_dic.items():
            u_1, u_2 = uu
            self.s_mat[u_1, u_2] = s[idx]
            self.s_mat[u_2, u_1] = s[idx]

    def fit_x(self, min_val=1, max_val=5, max_distance_to_rated=1, gamma=0., max_nfev_x=3,
              verbose_x=True, **kwargs):

        n_user, n_item = self._n_user, self._n_item

        new_x_mat = self.x_mat.copy()

        for it in tqdm(range(n_item), disable=not verbose_x, desc='GraphLearnerWNB:fit_x'):
            # Extract sub-matrices
            unrated_users_i = np.where(~self._ui_is_rated_mat[:, it])[0]
            rated_users_i = np.where(self._ui_is_rated_mat[:, it])[0]

            # Find the unrated users which are far from rated users
            rated_users_one_hot = self._ui_is_rated_mat[:, it] * 1
            adj_d_mat = self._adj_mat + self._adj_mat.T
            n_connected_rated = adj_d_mat.dot(rated_users_one_hot.reshape((-1, 1)))

            for _ in range(1, max_distance_to_rated):
                n_connected_rated = adj_d_mat.dot(n_connected_rated)

            far_to_be_updated_users = np.where(n_connected_rated == 0)[0]

            # Consider far users as rated
            unrated_users_i = np.array(list_minus_list(unrated_users_i, far_to_be_updated_users))
            rated_users_i = np.concatenate((rated_users_i, far_to_be_updated_users))

            n_unrated_users_i = len(unrated_users_i)
            if n_unrated_users_i == 0:
                continue

            s_minus_eye_mat = self.s_mat - np.eye(n_user)
            s_unrated_i_mat = s_minus_eye_mat[:, unrated_users_i]
            s_rated_i_mat = s_minus_eye_mat[:, rated_users_i]

            x_rated_i = self.x_mat[rated_users_i, it:(it + 1)]

            y_i = -s_rated_i_mat.dot(x_rated_i)

            # Add stability term
            y_i_extended = np.concatenate(
                (y_i, np.sqrt(gamma)*np.mean(self.x_mat[unrated_users_i], axis=1).reshape((-1, 1))),
                axis=0
            )
            s_unrated_i_extended_mat = \
                np.concatenate((s_unrated_i_mat, np.sqrt(gamma)*np.eye(n_unrated_users_i)), axis=0)

            # Least square estimation
            x_unrated_i = self.x_mat[unrated_users_i, it:(it + 1)]
            x_unrated_i_new = GraphLearner.ls(
                x_0=x_unrated_i[:, 0],
                s_unrated_mat=s_unrated_i_extended_mat,
                y=y_i_extended,
                min_val=min_val,
                max_val=max_val,
                max_nfev=max_nfev_x
            )

            # Update the x_mat
            new_x_mat[unrated_users_i, it] = x_unrated_i_new[:, 0]

        self.x_mat = new_x_mat

    def predict(self, x_mat, **kwargs):
        return self.s_mat.dot(x_mat)

    def save_to_file(self, savepath, filename, ext_dic=None):
        dic = {
            'x_mat': self.x_mat,
            's_mat': self.s_mat,
            'adj_mat': self._adj_mat,
            'ui_is_rated_mat': self._ui_is_rated_mat,
            'ext': ext_dic
        }

        with open(os.path.join(savepath, filename + '.graphlearnerwnb'), 'wb') as f:
            pickle.dump(dic, f)

    @staticmethod
    def load_from_file(loadpath, filename):
        with open(os.path.join(loadpath, filename + '.graphlearnerwnb'), 'rb') as f:
            dic = pickle.load(f)

        g_learner = GraphLearnerWithNoBias(adj_mat=dic['adj_mat'], ui_is_rated_mat=dic['ui_is_rated_mat'])

        g_learner.x_mat = dic['x_mat']
        g_learner.s_mat = dic['s_mat']

        return g_learner, dic


class CounterfactualGraphLearner(GraphLearnerBase):
    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

        self.s_mat_list = []  # list of sparse [(n_user + 1) x (n_user + 1)] matrices
        self.ks_mat = None  # [n_item x n_item]

    @staticmethod
    def from_graph_object(g, ui_is_rated_mat):
        # Instantiate a GraphLearner from graph
        g_learner = GraphLearner.from_graph_object(g, ui_is_rated_mat)

        # Extract the s_mat of GraphLearner
        csr_s_mat = csr_matrix(g_learner.s_mat)

        # Instantiate a CGLearner
        cg_learner = CounterfactualGraphLearner(g.adj_mat, ui_is_rated_mat)

        # Fill in the s_mat list
        for _ in range(cg_learner._n_item):
            cg_learner.s_mat_list.append(csr_s_mat.copy())

        return cg_learner

    def fit_ks_statistics(self, rat_mat):
        self.ks_mat = np.zeros((self._n_item, self._n_item))

        for it in tqdm(range(self._n_item)):
            self.ks_mat[it] = self._calc_ks_statistics_per_item(it, rat_mat)

    def _calc_ks_statistics_per_item(self, it, rat_mat):
        ks = np.zeros((self._n_item,))

        for o_it in range(self._n_item):
            if o_it == it:
                continue

            rated_vals_o_with_nan = rat_mat[self._ui_is_rated_mat[:, it], o_it]
            unrated_vals_o_with_nan = rat_mat[~self._ui_is_rated_mat[:, it], o_it]

            rated_vals_o = rated_vals_o_with_nan[~np.isnan(rated_vals_o_with_nan)]
            unrated_vals_o = unrated_vals_o_with_nan[~np.isnan(unrated_vals_o_with_nan)]

            if len(rated_vals_o) == 0 or len(unrated_vals_o) == 0:
                ks[o_it] = 0.5
                continue

            ks[o_it] = ks_2samp(rated_vals_o, unrated_vals_o).statistic / 2

        return ks

    def fit_shift_operator(self, l2_lambda_s=0., verbose_s=True, **kwargs):
        n_item = self._n_item

        for it in tqdm(range(n_item), disable=not verbose_s):
            ui_is_rated_wo_i_mat = self._ui_is_rated_mat.copy()
            ui_is_rated_wo_i_mat[:, it] = 0

            # Instantiate a GraphLearner
            g_learner_i = GraphLearner(self._adj_mat, ui_is_rated_wo_i_mat)
            g_learner_i.x_mat = self.x_mat

            # Fit the GraphLearner with corresponding weights
            g_learner_i.fit_shift_operator(l2_lambda_s=l2_lambda_s,
                                           weights=(1 - self.ks_mat[it]),
                                           verbose_s=False,
                                           **kwargs)

            # Fill the list of s_mats
            self.s_mat_list[it] = csr_matrix(g_learner_i.s_mat)

    def fit_x(self, min_val=1, max_val=5, max_distance_to_rated=1, l2_lambda_x=0., gamma=0.,
              verbose_x=True, **kwargs):

        n_item = self._n_item

        new_x_mat = self.x_mat.copy()

        for it in tqdm(range(n_item), disable=not verbose_x):
            g_learner_i = GraphLearner(adj_mat=self._adj_mat, ui_is_rated_mat=self._ui_is_rated_mat[:, it:(it + 1)])

            g_learner_i.s_mat = self.s_mat_list[it].toarray()
            g_learner_i.x_mat = self.x_mat[:, it:(it + 1)]

            g_learner_i.fit_x(min_val=min_val,
                              max_val=max_val,
                              max_distance_to_rated=max_distance_to_rated,
                              l2_lambda_x=l2_lambda_x,
                              gamma=gamma,
                              verbose_x=False,
                              **kwargs)

            new_x_mat[:, it:(it + 1)] = g_learner_i.x_mat

        self.x_mat = new_x_mat

    def predict(self, x_mat, **kwargs):
        x_pr_mat = np.zeros(x_mat.shape)

        for it in range(self._n_item):
            x_pr_mat[:, it:(it + 1)] = self.s_mat_list[it].dot(x_mat[:, it:(it + 1)])

        return x_pr_mat[:-1]

    def save_to_file(self, savepath, file_name, ext_dic=None):
        pass

    @staticmethod
    def load_from_file(loadpath, file_name):
        pass


class GraphMatrixCompletion(GraphLearnerBase):
    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

    @staticmethod
    def from_graph_object(g, ui_is_rated_mat):
        gmc = GraphMatrixCompletion(g.adj_mat, ui_is_rated_mat)

        g_learner = GraphLearner.from_graph_object(g, ui_is_rated_mat)

        gmc.s_mat = g_learner.s_mat

        return gmc

    def fit_shift_operator(self, **kwargs):
        g_learner = GraphLearner(adj_mat=self._adj_mat, ui_is_rated_mat=self._ui_is_rated_mat)
        g_learner.x_mat = self.x_mat
        g_learner.s_mat = self.s_mat

        g_learner.fit_shift_operator(**kwargs)

        self.s_mat = g_learner.s_mat

    def fit_x(self, beta=1, eps_x=1e-3, max_iter_x=10, verbose_x=True, **_kwargs):
        s_tilde_mat = (self.s_mat.T - np.eye(self._n_user + 1)).dot(self.s_mat - np.eye(self._n_user + 1))

        it = 1
        while True:
            if it == max_iter_x:
                break

            # Line-search
            t, p_k = self._line_search_for_t(self.x_mat[:-1], self.s_mat, s_tilde_mat)

            # Gradient descent
            g_k = np.concatenate((unvectorize(-p_k, n_row=self._n_user), np.zeros((1, self._n_item))), axis=0)
            x_new_mat = self.x_mat - t * g_k

            # Proximate
            x_prox_mat = self.proximate(x_new_mat, t, beta)

            # Project
            x_proj_mat = self.project(self.x_mat, x_prox_mat, self._ui_is_rated_mat)

            # Check the stopping criterion
            eps_x_k = np.linalg.norm(x_proj_mat - self.x_mat) / np.linalg.norm(self.x_mat)

            if verbose_x:
                print('it: %d, relative change of x is: %.3f' % (it, eps_x_k))

            self.x_mat = x_proj_mat

            if eps_x_k < eps_x:
                break

            it += 1

    @staticmethod
    def _line_search_for_t(x_mat, s_mat, s_tilde_mat):
        xk = vectorize(x_mat)
        pk = -GraphMatrixCompletion.jac_s2(xk, s_mat, s_tilde_mat)

        t = line_search(f=GraphMatrixCompletion.s2,
                        myfprime=GraphMatrixCompletion.jac_s2,
                        xk=xk,
                        pk=pk,
                        c1=0.01,
                        c2=0.9,
                        args=(s_mat, s_tilde_mat))[0]

        return t, pk

    @staticmethod
    def s2(x, s_mat, _s_tilde_mat):
        x_wo_ones_mat = unvectorize(x, s_mat.shape[0] - 1)
        x_mat = np.concatenate((x_wo_ones_mat, np.ones((1, x_wo_ones_mat.shape[1]))), axis=0)

        return np.linalg.norm((s_mat - np.eye(s_mat.shape[0])).dot(x_mat)) ** 2

    @staticmethod
    def jac_s2(x, _s_mat, s_tilde_mat):
        x_wo_ones_mat = unvectorize(x, s_tilde_mat.shape[0] - 1)
        x_mat = np.concatenate((x_wo_ones_mat, np.ones((1, x_wo_ones_mat.shape[1]))), axis=0)

        return vectorize(2 * s_tilde_mat.dot(x_mat)[:-1])

    @staticmethod
    def proximate(x_mat, t, beta):
        tau = t * beta

        rank_x = np.linalg.matrix_rank(x_mat)

        n_svd_comp = 3
        u_mat, s, vh_mat = randomized_svd(x_mat, n_components=n_svd_comp)
        while (s[-1] > tau) and (n_svd_comp < rank_x):
            n_svd_comp = np.max([2 * n_svd_comp, rank_x])
            u_mat, s, vh_mat = randomized_svd(x_mat, n_components=n_svd_comp)
        # ToDo
        # print('_proximate: %d components was enough' % n_svd_comp)

        s[s < tau] = 0
        s[s > tau] -= tau

        return u_mat.dot(np.diag(s)).dot(vh_mat)

    @staticmethod
    def project(x_mat_old, x_new_mat_org, ui_is_rated_mat):
        x_new_mat = x_mat_old.copy()

        x_new_mat[:-1][~ui_is_rated_mat] = x_new_mat_org[:-1][~ui_is_rated_mat]

        return x_new_mat

    def predict(self, x_mat, **kwargs):
        return self.s_mat.dot(x_mat)[:-1]

    def save_to_file(self, savepath, filename, ext_dic=None):
        dic = {
            'x_mat': self.x_mat,
            's_mat': self.s_mat,
            'adj_mat': self._adj_mat,
            'ui_is_rated_mat': self._ui_is_rated_mat,
            'ext': ext_dic
        }

        with open(os.path.join(savepath, filename + '.gmc'), 'wb') as f:
            pickle.dump(dic, f)

    @staticmethod
    def load_from_file(loadpath, filename):
        with open(os.path.join(loadpath, filename + '.gmc'), 'rb') as f:
            dic = pickle.load(f)

        gmc = GraphMatrixCompletion(adj_mat=dic['adj_mat'], ui_is_rated_mat=dic['ui_is_rated_mat'])

        gmc.x_mat = dic['x_mat']
        gmc.s_mat = dic['s_mat']

        return gmc, dic


class GraphMatrixCompletionWithNoBias(GraphLearnerBase):
    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

        self._g_learner = GraphLearnerWithNoBias(adj_mat, ui_is_rated_mat)

    @staticmethod
    def from_graph_object(g, ui_is_rated_mat):
        adj_mat = g.adj_mat
        w_mat = g.w_mat

        # Instantiate a GMC
        gmc = GraphMatrixCompletionWithNoBias(adj_mat, ui_is_rated_mat)

        # Find the largest eigenvalue
        w_1 = eigsh(w_mat, k=1, which='LM', return_eigenvectors=False)[0]

        # Set the shift operator as the normalized weight matrix
        gmc.s_mat = w_mat/w_1

        return gmc

    def fit_shift_operator(self, **kwargs):
        self._g_learner.x_mat = self.x_mat
        self._g_learner.s_mat = self.s_mat

        self._g_learner.fit_shift_operator(**kwargs)

        self.s_mat = self._g_learner.s_mat

    def fit_x(self, beta=1, eps_x=1e-3, max_iter_x=10, verbose_x=True, **_kwargs):
        s_tilde_mat = (self.s_mat.T - np.eye(self._n_user)).dot(self.s_mat - np.eye(self._n_user))

        it = 1
        while True:
            if it == max_iter_x:
                break

            # Line-search
            t, p_k = self._line_search_for_t(self.x_mat, self.s_mat, s_tilde_mat)

            # Gradient descent
            g_k = unvectorize(-p_k, n_row=self._n_user)
            x_new_mat = self.x_mat - t*g_k

            # Proximate
            x_prox_mat = self.proximate(x_new_mat, t, beta)

            # Project
            x_proj_mat = self.project(self.x_mat, x_prox_mat, self._ui_is_rated_mat)

            # Check the stopping criterion
            eps_x_k = np.linalg.norm(x_proj_mat - self.x_mat)/np.linalg.norm(self.x_mat)

            if verbose_x:
                print('it: %d, relative change of x is: %.3f' % (it, eps_x_k))

            self.x_mat = x_proj_mat

            if eps_x_k < eps_x:
                break

            it += 1

    @staticmethod
    def _line_search_for_t(x_mat, s_mat, s_tilde_mat):
        xk = vectorize(x_mat)
        pk = -GraphMatrixCompletionWithNoBias.jac_s2(xk, s_mat, s_tilde_mat)

        t = line_search(
            f=GraphMatrixCompletionWithNoBias.s2,
            myfprime=GraphMatrixCompletionWithNoBias.jac_s2,
            xk=xk,
            pk=pk,
            c1=0.01,
            c2=0.9,
            args=(s_mat, s_tilde_mat)
        )[0]

        return t, pk

    @staticmethod
    def s2(x, s_mat, _s_tilde_mat):
        x_mat = unvectorize(x, s_mat.shape[0])

        return np.linalg.norm((s_mat - np.eye(s_mat.shape[0])).dot(x_mat))**2

    @staticmethod
    def jac_s2(x, _s_mat, s_tilde_mat):
        x_mat = unvectorize(x, s_tilde_mat.shape[0])

        return vectorize(2*s_tilde_mat.dot(x_mat))

    @staticmethod
    def proximate(x_mat, t, beta):
        tau = t*beta

        rank_x = np.linalg.matrix_rank(x_mat)

        n_svd_comp = 3
        u_mat, s, vh_mat = randomized_svd(x_mat, n_components=n_svd_comp)
        while (s[-1] > tau) and (n_svd_comp < rank_x):
            n_svd_comp = np.max([2*n_svd_comp, rank_x])
            u_mat, s, vh_mat = randomized_svd(x_mat, n_components=n_svd_comp)
        # ToDo
        # print('_proximate: %d components was enough' % n_svd_comp)

        s[s < tau] = 0
        s[s > tau] -= tau

        return u_mat.dot(np.diag(s)).dot(vh_mat)

    @staticmethod
    def project(x_mat_old, x_new_mat_org, ui_is_rated_mat):
        x_new_mat = x_mat_old.copy()

        x_new_mat[~ui_is_rated_mat] = x_new_mat_org[~ui_is_rated_mat]

        return x_new_mat

    def predict(self, x_mat, **kwargs):
        return self.s_mat.dot(x_mat)

    def save_to_file(self, savepath, filename, ext_dic=None):
        dic = {
            'x_mat': self.x_mat,
            's_mat': self.s_mat,
            'adj_mat': self._adj_mat,
            'ui_is_rated_mat': self._ui_is_rated_mat,
            'ext': ext_dic
        }

        with open(os.path.join(savepath, filename + '.gmcwnb'), 'wb') as f:
            pickle.dump(dic, f)

    @staticmethod
    def load_from_file(loadpath, filename):
        with open(os.path.join(loadpath, filename + '.gmcwnb'), 'rb') as f:
            dic = pickle.load(f)

        gmc = GraphMatrixCompletionWithNoBias(adj_mat=dic['adj_mat'], ui_is_rated_mat=dic['ui_is_rated_mat'])

        gmc.x_mat = dic['x_mat']
        gmc.s_mat = dic['s_mat']

        return gmc, dic


class CounterfactualGraphMatrixCompletion(GraphLearnerBase):
    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

        # ToDo: self.s_mat?

        self.cg_learner = CounterfactualGraphLearner(adj_mat, ui_is_rated_mat)

    @staticmethod
    def from_graph_object(g: Graph, ui_is_rated_mat):
        # Instantiate a CGLearner
        cg_learner = CounterfactualGraphLearner.from_graph_object(g, ui_is_rated_mat)

        # Init. a CGMC
        cgmc = CounterfactualGraphMatrixCompletion(adj_mat=g.adj_mat, ui_is_rated_mat=ui_is_rated_mat)

        # Fill in the CGLearner
        cgmc.cg_learner = cg_learner

        return cgmc

    def fit_ks_statistics(self, rat_mat):
        self.cg_learner.fit_ks_statistics(rat_mat)

    def fit_shift_operator(self, l2_lambda_s=0., verbose_s=True, **kwargs):
        self.cg_learner.fit_shift_operator(l2_lambda_s=l2_lambda_s, verbose_s=verbose_s, **kwargs)

    def fit_x(self, beta=1, eps_x=1e-3, verbose_x=True, **kwargs):
        s_tilde_mat_list = []
        for s_mat in self.cg_learner.s_mat_list:
            s_tilde_mat = (s_mat.T - np.eye(self._n_user + 1)).dot(s_mat - np.eye(self._n_user + 1))
            s_tilde_mat_list.append(s_tilde_mat)

        it = 1
        max_it = 50
        while True:
            if it == max_it:
                break
            if verbose_x:
                print('fitting x --> iter: %d' % it)

            # Line-search
            t, p_k = self._line_search_for_t(self.x_mat[:-1], self.cg_learner.s_mat_list, s_tilde_mat_list)

            # Gradient descent
            g_k = np.concatenate((unvectorize(-p_k, n_row=self._n_user), np.zeros((1, self._n_item))), axis=0)
            x_new_mat = self.x_mat - t * g_k

            # Proximate
            x_prox_mat = GraphMatrixCompletion.proximate(x_new_mat, t, beta)

            # Project
            x_proj_mat = GraphMatrixCompletion.project(self.x_mat, x_prox_mat, self._ui_is_rated_mat)

            # Check the stopping criterion
            eps_x_k = np.linalg.norm(x_proj_mat - self.x_mat) / np.linalg.norm(self.x_mat)

            self.cg_learner.x_mat = x_proj_mat
            self.x_mat = x_proj_mat

            if eps_x_k < eps_x:
                break
            else:
                if verbose_x:
                    print('relative change of x is: %.3f' % eps_x_k)

            it += 1

    @staticmethod
    def s2(x, s_mat_list, _s_tilde_mat_list):
        x_wo_ones_mat = unvectorize(x, s_mat_list[0].shape[0] - 1)

        loss = 0
        for i_s, s_mat in enumerate(s_mat_list):
            loss += GraphMatrixCompletion.s2(x_wo_ones_mat[:, i_s], s_mat, _s_tilde_mat_list[i_s])

        return loss

    @staticmethod
    def jac_s2(x, _s_mat_list, s_tilde_mat_list):
        x_wo_ones_mat = unvectorize(x, s_tilde_mat_list[0].shape[0] - 1)

        jac_mat = np.zeros(x_wo_ones_mat.shape)
        for it in range(jac_mat.shape[1]):
            jac_mat[:, it] = GraphMatrixCompletion.jac_s2(x_wo_ones_mat[:, it], _s_mat_list[it], s_tilde_mat_list[it])

        return vectorize(jac_mat)

    @staticmethod
    def _line_search_for_t(x_mat, s_mat_list, s_tilde_mat_list):
        xk = vectorize(x_mat)
        pk = -CounterfactualGraphMatrixCompletion.jac_s2(xk, s_mat_list, s_tilde_mat_list)

        t = line_search(f=CounterfactualGraphMatrixCompletion.s2,
                        myfprime=CounterfactualGraphMatrixCompletion.jac_s2,
                        xk=xk,
                        pk=pk,
                        c1=0.01,
                        c2=0.9,
                        args=(s_mat_list, s_tilde_mat_list))[0]

        return t, pk

    def predict(self, x_mat, **kwargs):
        return self.cg_learner.predict(x_mat, **kwargs)

    def save_to_file(self, savepath, file_name, ext_dic=None):
        # ToDo
        pass

    @staticmethod
    def load_from_file(loadpath, file_name):
        # ToDo
        pass


if __name__ == '__main__':
    adj_mat_ = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    t_0_ = np.random.random((1, 200)) - 0.5
    t_2_ = np.random.random((1, 200)) - 0.5
    rating_mat_ = np.concatenate((t_0_, t_0_ + t_2_ + np.random.random((1, 200))*0.1, t_2_), axis=0)

    n_u_, n_i_ = rating_mat_.shape

    gl_ = GraphLearnerWithNoBias(adj_mat_, ~np.isnan(rating_mat_))

    # x_mat_ = np.concatenate((rating_mat_, np.ones((1, n_i_))), axis=0)
    x_mat_ = rating_mat_

    gl_.x_mat = x_mat_
    # gl_.s_mat = np.zeros((n_u_ + 1, n_u_ + 1))
    gl_.s_mat = np.zeros((n_u_, n_u_))

    gl_.fit_shift_operator()
