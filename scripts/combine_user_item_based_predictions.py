import os
import numpy as np
from sklearn.linear_model import LinearRegression

from app.transformers.graph import Graph
from app.models.graphlearning import GraphLearner, GraphMatrixCompletion
from app.utils.log import Logger
from app.utils.data_handler import load_dataset
from app.utils.mathtools import rmse

if __name__ == '__main__':
    # ----- Settings -----
    dataset_sett = {}
    graph_sett = {}
    g_learner_sett = {}

    # Path
    data_load_path = os.path.join('..', 'data', 'ml-100k')
    g_learner_load_path = os.path.join('..', 'results', 'graphlearners')
    graph_load_path = os.path.join('..', 'results', 'graphs')

    # Dataset
    dataset_sett['dataset_name'] = 'ml-100k'
    dataset_sett['part'] = 3

    # Graph
    graph_sett['min_num_common_items'] = 8
    graph_sett['max_degree'] = 3

    # GraphLearner
    g_learner_sett['max_distance_to_rated'] = 1
    g_learner_sett['gamma'] = 1
    g_learner_sett['max_nfev_x'] = 5
    # g_learner_sett['beta'] = 100
    # g_learner_sett['eps_x'] = 1e-2
    g_learner_sett['l2_lambda_s'] = 1

    # User-based
    dataset_sett['do_transpose'] = False

    u_sett = g_learner_sett.copy()
    u_sett.update(graph_sett)
    u_sett.update(dataset_sett)

    # Item-based
    dataset_sett['do_transpose'] = True

    i_sett = g_learner_sett.copy()
    i_sett.update(graph_sett)
    i_sett.update(dataset_sett)

    # Select class
    g_learner_class = GraphLearner

    # ----- Load graph -----
    print('Loading graph ...')
    dataset_sett['do_transpose'] = False
    graph_load_sett = graph_sett.copy()
    graph_load_sett.update(dataset_sett)
    _, graph_dic = Graph.load_from_file(load_path=graph_load_path,
                                        file_name='graph' + Logger.stringify(graph_load_sett))

    # ----- Load data -----
    print('Loading data ...')
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item = \
        load_dataset(data_load_path, **graph_dic['ext']['dataset'])

    # ----- Load GraphLearners -----
    u_g_learner, _ = g_learner_class.load_from_file(loadpath=g_learner_load_path,
                                                    filename='graphlearner' + Logger.stringify(u_sett))

    i_g_learner, _ = g_learner_class.load_from_file(loadpath=g_learner_load_path,
                                                    filename='graphlearner' + Logger.stringify(i_sett))

    # ----- Eval. user/item-based methods -----
    u_rmse_va = rmse(rating_mat_va, u_g_learner.x_mat[:-1])
    u_rmse_te = rmse(rating_mat_te, u_g_learner.x_mat[:-1])

    i_rmse_va = rmse(rating_mat_va.T, i_g_learner.x_mat[:-1])
    i_rmse_te = rmse(rating_mat_te.T, i_g_learner.x_mat[:-1])

    print('User-based performance --> rmse val: %.3f rmse test: %.3f' % (u_rmse_va, u_rmse_te))
    print('Item-based performance --> rmse val: %.3f rmse test: %.3f' % (i_rmse_va, i_rmse_te))

    # ----- Combine -----
    reg = LinearRegression()

    # Format val. data
    mask_va = ~np.isnan(rating_mat_va)

    x_va = np.concatenate((u_g_learner.x_mat[:-1][mask_va].reshape((-1, 1)),
                           i_g_learner.x_mat[:-1].T[mask_va].reshape((-1, 1))), axis=1)
    y_va = rating_mat_va[mask_va]

    # Fit a regression model on the val. data
    reg.fit(x_va, y_va)

    # Format test data
    mask_te = ~np.isnan(rating_mat_te)

    x_te = np.concatenate((u_g_learner.x_mat[:-1][mask_te].reshape((-1, 1)),
                           i_g_learner.x_mat[:-1].T[mask_te].reshape((-1, 1))), axis=1)
    y_te = rating_mat_te[mask_te]

    # Predict on the test data
    y_pr = reg.predict(x_te)
    y_pr[y_pr > graph_dic['ext']['dataset']['max_value']] = graph_dic['ext']['dataset']['max_value']
    y_pr[y_pr < graph_dic['ext']['dataset']['min_value']] = graph_dic['ext']['dataset']['min_value']

    # ----- Eval combined -----
    print('Combined performance --> rmse: %.3f' % rmse(y_te, y_pr))
