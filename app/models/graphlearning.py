import os
from tqdm import tqdm
import pickle
import numpy as np
from numpy.random import default_rng
from scipy.optimize import least_squares
from matplotlib import pyplot as plt

from app.models.model_base import Model
from app.transformers.graph import Graph
from app.utils.mathtools import ls, rmse, list_minus_list
from app.utils.log import Logger


class GraphLearner(Model):
    def __init__(self, adj_mat, ui_is_rated_mat):
        self._adj_mat = adj_mat  # [n_user x n_user]
        self._ui_is_rated_mat = ui_is_rated_mat  # [n_user x n_item]
        self._n_user, self._n_item = ui_is_rated_mat.shape

        self.x_mat = None  # [(n_user + 1) x n_item]
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

    def fit_shift_operator(self, l2_lambda=0.):
        n_user = self._n_user

        self.s_mat = np.zeros((n_user + 1, n_user + 1))
        self.s_mat[n_user, n_user] = 1

        for u in tqdm(range(n_user)):
            # Extract sub-matrices
            users_u = np.where(self._adj_mat[u] == 1)[0]
            idx_s_u = np.concatenate((users_u, [n_user]))

            items_u = np.where(self._ui_is_rated_mat[u])[0]

            x_mat_u = self.x_mat[idx_s_u][:, items_u]

            x_u = self.x_mat[u, items_u]

            # Least square estimation
            s_u = ls(x_mat_u.T, x_u, l2_lambda=l2_lambda)

            # Fill the s_mat
            self.s_mat[u, idx_s_u] = s_u[:, 0]

    def fit_x(self, min_val=1, max_val=5, max_distance_to_rated=1, l2_lambda=0., gamma=0.):
        n_user, n_item = self._n_user, self._n_item

        new_x_mat = self.x_mat.copy()

        for it in tqdm(range(n_item)):
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
            # x_unrated_i_new = ls(s_unrated_i_extended_mat, y_i_extended, l2_lambda=l2_lambda)
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

    def fit(self, x_0_mat, **kwargs):
        self.x_mat = x_0_mat

        verbose = kwargs['verbose']

        if verbose:
            print('Updating x_mat ...')
        self.fit_x(min_val=kwargs['min_val'],
                   max_val=kwargs['max_val'],
                   max_distance_to_rated=kwargs['max_distance_to_rated'],
                   l2_lambda=kwargs['l2_lambda_x'],
                   gamma=kwargs['gamma'])

        if verbose:
            print('Updating s_mat ...')
        self.fit_shift_operator(l2_lambda=kwargs['l2_lambda_s'])

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


def plot_users(rat_tr, rat_te, rat_pr, users):
    n_user = len(users)

    for i_u, u in enumerate(users):
        plt.subplot(n_user, 1, i_u + 1)

        plt.plot(rat_te[u], rat_pr[u], 'r*')
        plt.plot(rat_tr[u], rat_pr[u], 'k*')
        plt.plot([np.min(rat_pr[u]), np.max(rat_pr[u])], [np.min(rat_pr[u]), np.max(rat_pr[u])], 'k--')

    plt.show()


def simulate_sample_ratings(n_item, min_val, max_val, sigma_n, p_miss):
    rng = default_rng(0)

    # Simulate raw ratings
    r1 = rng.integers(low=min_val, high=max_val, size=(1, n_item)).astype(float)
    rat_mat_1 = np.concatenate((r1, 2*r1 - 3, -2*r1 + 5), axis=0)

    r2 = rng.integers(low=min_val, high=max_val, size=(1, n_item)).astype(float)
    rat_mat_2 = np.concatenate((r2, 2*r2 - 4, -2*r2 + 6), axis=0)

    rat_mat_all = np.concatenate((rat_mat_1, rat_mat_2), axis=0)
    rat_mat_all += rng.normal(loc=0, scale=sigma_n, size=rat_mat_all.shape)

    # Remove random elements
    mask_nan = rng.random(size=rat_mat_all.shape) < p_miss

    rat_mat_o = rat_mat_all.copy()
    rat_mat_o[mask_nan] = np.nan

    rat_mat_m = rat_mat_all.copy()
    rat_mat_m[~mask_nan] = np.nan

    return rat_mat_o, rat_mat_m


if __name__ == '__main__':
    # ----- Settings -----
    sett = {}
    # General
    n_iter = 10
    do_plot_performance = False
    verbose_g_learner = True
    do_plot_ratings = True
    wait_time_to_plot_ratings = 0.5

    # Path
    save_path = os.path.join('results')
    os.makedirs(save_path, exist_ok=True)

    # Simulation
    num_item = 500
    min_value = 1
    max_value = 5
    sett['sigma_noise'] = 1
    sett['prob_miss'] = 0.9

    # Graph
    sett['min_num_common_items'] = 3
    sett['max_degree'] = 1

    # Learner
    sett['max_distance_to_rated'] = 2
    sett['l2_lambda_s'] = 0
    sett['l2_lambda_x'] = 0
    sett['gamma'] = 0.1

    # ----- Simulate sample ratings -----
    rat_mat_obs, rat_mat_missed = \
        simulate_sample_ratings(num_item, min_value, max_value, sett['sigma_noise'], sett['prob_miss'])

    # ----- Find the graph structure -----
    graph = Graph(min_num_common_items=sett['min_num_common_items'], max_degree=sett['max_degree'])
    graph.fit_transform(rat_mat_obs)

    # ----- Init. -----
    # x
    x_mat_filled_mean = np.concatenate((rat_mat_obs, np.ones((1, num_item))), axis=0)
    for user in range(rat_mat_obs.shape[0]):
        x_mat_filled_mean[user, np.isnan(x_mat_filled_mean[user])] = np.nanmean(rat_mat_obs[user])

    # Graph learner
    user_item_is_rated_mat = ~np.isnan(rat_mat_obs)
    graph_learner = GraphLearner.from_graph_object(graph, user_item_is_rated_mat)

    # Logger
    logger = Logger(settings=sett, save_path=save_path, do_plot=do_plot_performance)

    # ----- Big loop! -----
    x_0_matrix = x_mat_filled_mean.copy()

    rat_mat_pr = graph_learner.predict(x_0_matrix)
    rmse_tr = rmse(rat_mat_obs, rat_mat_pr)
    rmse_te = rmse(rat_mat_missed, rat_mat_pr)
    logger.log(rmse_tr, 0, rmse_te)

    if do_plot_ratings:
        plt.figure()
        plot_users(rat_mat_obs, rat_mat_missed, rat_mat_pr, [0, 1, 2])
        plt.pause(wait_time_to_plot_ratings)

    for iteration in range(n_iter):
        # Fit
        graph_learner.fit(x_0_matrix,
                          min_val=-10,
                          max_val=10,
                          max_distance_to_rated=sett['max_distance_to_rated'],
                          l2_lambda_x=sett['l2_lambda_x'],
                          gamma=sett['gamma'],
                          l2_lambda_s=sett['l2_lambda_s'],
                          verbose=verbose_g_learner)

        # Predict
        rat_mat_pr = graph_learner.predict(graph_learner.x_mat)
        rmse_tr = rmse(rat_mat_obs, rat_mat_pr)
        rmse_te = rmse(rat_mat_missed, rat_mat_pr)
        logger.log(rmse_tr, 0, rmse_te)

        # Update x
        x_0_matrix = graph_learner.x_mat

        # Plotting
        if do_plot_ratings:
            plt.clf()
            plot_users(rat_mat_obs, rat_mat_missed, rat_mat_pr, [0, 1, 2])
            plt.pause(wait_time_to_plot_ratings)
