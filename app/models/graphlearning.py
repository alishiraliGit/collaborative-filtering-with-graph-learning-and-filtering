import os
import abc
from tqdm import tqdm
import pickle
import numpy as np
from scipy.optimize import least_squares, line_search
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp
from sklearn.utils.extmath import randomized_svd

from app.models.model_base import Model
from app.transformers.graph import Graph
from app.utils.mathtools import ls, list_minus_list, vectorize, unvectorize


class GraphLearnerBase(Model, abc.ABC):
    def __init__(self, adj_mat, ui_is_rated_mat):
        self._adj_mat = adj_mat  # [n_user x n_user]
        self._ui_is_rated_mat = ui_is_rated_mat  # [n_user x n_item]
        self._n_user, self._n_item = ui_is_rated_mat.shape

        self.x_mat = None  # [(n_user + 1) x n_item]

    @staticmethod
    def from_graph_object(g: Graph, ui_is_rated_mat):
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


class GraphLearner(GraphLearnerBase):

    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

        self.s_mat = None  # [(n_user + 1) x (n_user + 1)]

    @staticmethod
    def from_graph_object(g: Graph, ui_is_rated_mat):
        # Instantiate a GraphLearner
        adj_mat = g.adj_mat
        g_learner = GraphLearner(adj_mat, ui_is_rated_mat)

        # Init. s_mat from graph w & b
        n_user = g.n_user
        s_mat = np.zeros((n_user + 1, n_user + 1))
        s_mat[n_user, n_user] = 1

        n_in_edge = np.sum(adj_mat, axis=1)

        b_normalized = np.sum(g.b_mat, axis=1)/n_in_edge
        w_normalized_mat = np.diag(1/n_in_edge).dot(g.w_mat)

        s_mat[:n_user, :n_user] = w_normalized_mat
        s_mat[:n_user, n_user] = b_normalized

        g_learner.s_mat = s_mat

        return g_learner

    def fit_shift_operator(self, l2_lambda_s=0., weights=np.array([1]), verbose_s=True, **kwargs):
        n_user = self._n_user

        self.s_mat = np.zeros((n_user + 1, n_user + 1))
        self.s_mat[n_user, n_user] = 1

        for u in tqdm(range(n_user), disable=not verbose_s):
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

    def fit_x(self, min_val=1, max_val=5, max_distance_to_rated=1, l2_lambda_x=0., gamma=0.,
              verbose_x=True, **kwargs):

        n_user, n_item = self._n_user, self._n_item

        new_x_mat = self.x_mat.copy()

        for it in tqdm(range(n_item), disable=not verbose_x):
            # Extract sub-matrices
            unrated_users_i = np.where(~self._ui_is_rated_mat[:, it])[0]
            rated_users_wo_bias_i = np.where(self._ui_is_rated_mat[:, it])[0]
            rated_users_i = np.concatenate((rated_users_wo_bias_i, [n_user]))

            # Find the unrated users which are far from rated users
            rated_users_one_hot = self._ui_is_rated_mat[:, it]*1
            adj_d_mat = self._adj_mat + self._adj_mat.T + np.eye(n_user)
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
                (y_i, np.sqrt(gamma)*np.mean(self.x_mat[unrated_users_i], axis=1).reshape((-1, 1))),
                axis=0
            )
            s_unrated_i_extended_mat = \
                np.concatenate((s_unrated_i_mat, np.sqrt(gamma)*np.eye(n_unrated_users_i)), axis=0)

            # Least square estimation
            # x_unrated_i_new = ls(s_unrated_i_extended_mat, y_i_extended, l2_lambda=l2_lambda_x)
            x_unrated_i = self.x_mat[unrated_users_i, it:(it + 1)]
            x_unrated_i_new = self.ls(x_0=x_unrated_i[:, 0],
                                      s_unrated_mat=s_unrated_i_extended_mat,
                                      y=y_i_extended,
                                      min_val=min_val,
                                      max_val=max_val)

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
    def ls(x_0, s_unrated_mat, y, min_val, max_val,
           method='trf', loss='linear', f_scale=1, max_nfev=3, verbose=0, **kwargs):

        res = least_squares(fun=GraphLearner.ls_fun,
                            jac=GraphLearner.ls_jac,
                            x0=x_0,
                            args=(s_unrated_mat, y),
                            bounds=(min_val, max_val),
                            method=method,
                            loss=loss,
                            f_scale=f_scale,
                            max_nfev=max_nfev,
                            verbose=verbose,
                            **kwargs)

        return res.x.reshape((-1, 1))

    def predict(self, x_mat, **kwargs):
        return self.s_mat.dot(x_mat)[:-1]

    def save_to_file(self, savepath, file_name, ext_dic=None):
        dic = {
            'x_mat': self.x_mat,
            's_mat': self.s_mat,
            'adj_mat': self._adj_mat,
            'ui_is_rated_mat': self._ui_is_rated_mat,
            'ext': ext_dic
        }

        with open(os.path.join(savepath, file_name + '.graphlearner'), 'wb') as f:
            pickle.dump(dic, f)

    @staticmethod
    def load_from_file(loadpath, file_name):
        with open(os.path.join(loadpath, file_name + '.graphlearner'), 'rb') as f:
            dic = pickle.load(f)

        g_learner = GraphLearner(adj_mat=dic['adj_mat'], ui_is_rated_mat=dic['ui_is_rated_mat'])

        g_learner.x_mat = dic['x_mat']
        g_learner.s_mat = dic['s_mat']

        return g_learner, dic


class CounterfactualGraphLearner(GraphLearnerBase):
    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

        self.s_mat_list = []  # list of sparse [(n_user + 1) x (n_user + 1)] matrices
        self.ks_mat = None  # [n_item x n_item]

    @staticmethod
    def from_graph_object(g: Graph, ui_is_rated_mat):
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

            ks[o_it] = ks_2samp(rated_vals_o, unrated_vals_o).statistic/2

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

        self.s_mat = None  # [(n_user + 1) x (n_user + 1)]

        self.s_tilde_mat = None

    @staticmethod
    def from_graph_object(g: Graph, ui_is_rated_mat):
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

    def fit_x(self, beta=1, eps_x=1e-3, verbose_x=True, **kwargs):
        s_tilde_mat = (self.s_mat.T - np.eye(self._n_user + 1)).dot(self.s_mat - np.eye(self._n_user + 1))

        it = 1
        max_it = 50
        while True:
            if it == max_it:
                break
            if verbose_x:
                print('iter: %d' % it)

            # Line-search
            t, p_k = self._line_search_for_t(self.x_mat[:-1], self.s_mat, s_tilde_mat)

            # Gradient descent
            g_k = np.concatenate((unvectorize(-p_k, n_row=self._n_user), np.zeros((1, self._n_item))), axis=0)
            x_new_mat = self.x_mat - t*g_k

            # Proximate
            x_prox_mat = self.proximate(x_new_mat, t, beta)

            # Project
            x_proj_mat = self.project(self.x_mat, x_prox_mat, self._ui_is_rated_mat)

            # Check the stopping criterion
            eps_x_k = np.linalg.norm(x_proj_mat - self.x_mat)/np.linalg.norm(self.x_mat)
            if eps_x_k < eps_x:
                self.x_mat = x_proj_mat
                break
            else:
                if verbose_x:
                    print('relative change of x is: %.3f' % eps_x_k)
                self.x_mat = x_proj_mat

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

        return np.linalg.norm((s_mat - np.eye(s_mat.shape[0])).dot(x_mat))**2

    @staticmethod
    def jac_s2(x, _s_mat, s_tilde_mat):
        x_wo_ones_mat = unvectorize(x, s_tilde_mat.shape[0] - 1)
        x_mat = np.concatenate((x_wo_ones_mat, np.ones((1, x_wo_ones_mat.shape[1]))), axis=0)

        return vectorize(2*s_tilde_mat.dot(x_mat)[:-1])

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

        x_new_mat[:-1][~ui_is_rated_mat] = x_new_mat_org[:-1][~ui_is_rated_mat]

        return x_new_mat

    def predict(self, x_mat, **kwargs):
        return self.s_mat.dot(x_mat)[:-1]

    def save_to_file(self, savepath, file_name, ext_dic=None):
        # ToDo
        pass

    @staticmethod
    def load_from_file(loadpath, file_name):
        # ToDo
        pass


class CounterfactualGraphMatrixCompletion(GraphLearnerBase):
    def __init__(self, adj_mat, ui_is_rated_mat):
        super().__init__(adj_mat, ui_is_rated_mat)

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
            x_new_mat = self.x_mat - t*g_k

            # Proximate
            x_prox_mat = GraphMatrixCompletion.proximate(x_new_mat, t, beta)

            # Project
            x_proj_mat = GraphMatrixCompletion.project(self.x_mat, x_prox_mat, self._ui_is_rated_mat)

            # Check the stopping criterion
            eps_x_k = np.linalg.norm(x_proj_mat - self.x_mat)/np.linalg.norm(self.x_mat)

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
