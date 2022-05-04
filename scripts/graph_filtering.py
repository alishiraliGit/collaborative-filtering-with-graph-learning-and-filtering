import os
import numpy as np
from matplotlib import pyplot as plt

from app.transformers.graph import Graph
from app.transformers.graphfilter import GraphFilter
from app.models.graphlearning import GraphLearner, GraphMatrixCompletion
from app.utils.log import Logger
from app.utils.data_handler import load_dataset

if __name__ == '__main__':
    # ----- Settings -----
    dataset_sett = {}
    graph_sett = {}
    g_learner_sett = {}
    g_filter_sett = {}

    # General

    # Path
    data_load_path = os.path.join('..', 'data', 'ml-100k')
    graph_load_path = os.path.join('..', 'results', 'graphs')
    g_learner_load_path = os.path.join('..', 'results', 'graphlearners')

    save_path = os.path.join('..', 'results', 'freqanalysis')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_sett['dataset_name'] = 'ml-100k'
    dataset_sett['part'] = 3
    dataset_sett['do_transpose'] = False

    # Graph
    graph_sett['min_num_common_items'] = 8
    graph_sett['max_degree'] = 3

    # GraphLearner

    # Settings for GraphLearner class:
    # g_learner_sett['max_distance_to_rated'] = 1
    # g_learner_sett['gamma'] = 10
    # g_learner_sett['max_nfev_x'] = 10

    # Settings for GraphMatrixCompletion class:
    g_learner_sett['beta'] = 100
    g_learner_sett['eps_x'] = 1e-2
    g_learner_sett['l2_lambda_s'] = 10

    # GraphFilter
    g_filter_sett['bw'] = 0.3

    # ----- Load graph -----
    print('Loading graph ...')
    graph_load_sett = graph_sett.copy()
    graph_load_sett.update(dataset_sett)
    graph, graph_dic = Graph.load_from_file(
        load_path=graph_load_path,
        file_name='graph' + Logger.stringify(graph_load_sett)
    )

    # ------- Load data -------
    print('Loading data ...')
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item = \
        load_dataset(data_load_path, **graph_dic['ext']['dataset'])

    # ----- Load graph learner -----
    print('Loading graph learner ...')
    g_lerner_load_sett = g_learner_sett.copy()
    g_lerner_load_sett.update(graph_sett)
    g_lerner_load_sett.update(dataset_sett)

    graph_learner, g_learner_dic = GraphMatrixCompletion.load_from_file(
        loadpath=g_learner_load_path,
        filename='graphlearner' + Logger.stringify(g_lerner_load_sett)
    )

    # ----- Plot ratings -----
    rating_mat_pr = graph_learner.x_mat[:-1]

    mask_tr = ~np.isnan(rating_mat_tr)
    mask_te = ~np.isnan(rating_mat_te)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rating_mat_tr[mask_tr], rating_mat_pr[mask_tr], 'k*')
    plt.xlabel('train')
    plt.ylabel('prediction')

    plt.subplot(1, 2, 2)
    plt.plot(rating_mat_te[mask_te], rating_mat_pr[mask_te], 'k*')
    plt.xlabel('test')
    plt.ylabel('prediction')

    plt.axis('equal')

    # ----- Get l/h freq. shift ops -----
    print('Fit l/h freq. shift ops ...')
    graph_filter = GraphFilter(graph_learner.s_mat)

    graph_filter.fit(bw=g_filter_sett['bw'])
    x_low_mat, x_high_mat = graph_filter.transform(graph_learner.x_mat)

    rating_mat_low = x_low_mat[:-1]

    # Plot
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rating_mat_tr[mask_tr], rating_mat_low[mask_tr], 'r*')
    plt.xlabel('train')
    plt.ylabel('prediction')

    plt.subplot(1, 2, 2)
    plt.plot(rating_mat_te[mask_te], rating_mat_low[mask_te], 'r*')
    plt.xlabel('test')
    plt.ylabel('prediction')

    # Freq. response
    plt.figure()

    x_f = np.abs(graph_filter.v_inv_mat.dot(graph_learner.x_mat[:, 0:1]))
    plt.plot(np.abs(graph_filter.v_inv_mat.dot(graph_learner.x_mat[:, 0:1])), 'k')
    plt.plot(np.abs(graph_filter.v_inv_mat.dot(x_low_mat[:, 0:1])), 'r')
