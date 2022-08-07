import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from app.transformers.graph import SmoothGraph
from app.models.graphlearning import GraphLearner, CounterfactualGraphLearner, GraphMatrixCompletion, \
    CounterfactualGraphMatrixCompletion, GraphLearnerWithNoBias, GraphMatrixCompletionWithNoBias
from app.utils.mathtools import fill_with_row_means
from app.utils.log import Logger
from app.utils.data_handler import load_dataset
from core.coordinatedescent import GraphLearningCD

if __name__ == '__main__':
    # ----- Settings -----
    dataset_sett = {}
    ext_graph_sett = {}
    g_learner_sett = {}

    # General
    do_plot_performance_while_logging = False
    do_save = False

    # Path
    data_load_path = os.path.join('..', 'data', 'ml-100k')
    ext_graph_load_path = os.path.join('..', 'results', 'graphs', 'ext')

    save_path = os.path.join('..', 'results', 'graphlearners')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_sett['name'] = 'ml-100k'
    dataset_sett['part'] = 3
    dataset_sett['min_value'] = 1
    dataset_sett['max_value'] = 5
    dataset_sett['do_transpose'] = False

    # Cross-validation
    dataset_sett['va_split'] = 0.1
    dataset_sett['random_state'] = 1

    # Graph
    ext_graph_sett['name'] = 'graph_a0_b-2'

    # GraphLearner

    # Settings for GraphLearner class:
    # g_learner_sett['max_distance_to_rated'] = 1
    # g_learner_sett['gamma'] = 10
    # g_learner_sett['max_nfev_x'] = 10
    # g_learner_sett['l2_lambda_s'] = 10

    # Settings for GraphMatrixCompletion class:
    g_learner_sett['beta'] = 100  # 100
    g_learner_sett['eps_x'] = 1e-3  # 1e-2, 1e-3
    g_learner_sett['max_iter_x'] = 20  # 20
    g_learner_sett['l2_lambda_s'] = 10  # 10
    g_learner_sett['l1_ratio_s'] = 0  # 0
    g_learner_sett['max_iter_s'] = 20  # 1 for bvls, 20 for trf
    g_learner_sett['bound_s'] = False

    verbose_x = False
    verbose_s = True

    # Coordinate Descent
    n_iter = 6
    calc_bias = False
    verbose_cd = True

    # ----- Load ext graph -----
    print('Loading ext graph ...')
    with open(os.path.join(ext_graph_load_path, ext_graph_sett['name'] + '.pkl'), 'rb') as f:
        ext_graph_dic = pickle.load(f)

    graph = SmoothGraph(None, None, None)

    graph.adj_mat = ext_graph_dic['adj_mat']
    graph.w_mat = ext_graph_dic['w_mat']
    graph.n_user = graph.adj_mat.shape[0]

    # ------- Load data -------
    print('Loading data ...')
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item = \
        load_dataset(data_load_path, **dataset_sett)

    # ----- Init. -----
    print('Initializing ...')
    # x_mat
    # With bias:
    # x_0_matrix = fill_with_row_means(np.concatenate((rating_mat_tr, np.ones((1, n_item))), axis=0))
    # Without bias:
    x_0_matrix = fill_with_row_means(rating_mat_tr)

    # Graph learner
    user_item_is_rated_mat = ~np.isnan(rating_mat_tr)

    graph_learner = GraphMatrixCompletionWithNoBias.from_graph_object(graph, user_item_is_rated_mat)

    # Fit the ks stats
    # print('Fitting KS-statistics ...')
    # graph_learner.fit_ks_statistics(rat_mat=rating_mat_tr)

    # Logger
    log_sett = g_learner_sett.copy()
    log_sett.update(ext_graph_sett)
    log_sett.update(dataset_sett)
    logger_x = Logger(settings=log_sett, save_path=save_path, do_plot=do_plot_performance_while_logging, title='x')
    logger_s = Logger(settings=log_sett, save_path=save_path, do_plot=do_plot_performance_while_logging, title='Sx')

    # ----- Coordinate descent -----
    # ToDo
    plt.figure()
    plt.plot(graph.w_mat[graph.adj_mat == 1], graph_learner.s_mat[graph.adj_mat == 1], 'k*')

    cd = GraphLearningCD(
        g_learner=graph_learner,
        logger_x=logger_x,
        logger_s=logger_s
    )

    cd.run(
        x_0_mat=x_0_matrix,
        rat_mat_tr=rating_mat_tr,
        rat_mat_va=rating_mat_va,
        rat_mat_te=rating_mat_te,
        n_iter=n_iter,
        verbose=verbose_cd,
        min_val=dataset_sett['min_value'],
        max_val=dataset_sett['max_value'],
        calc_bias=calc_bias,
        verbose_x=verbose_x,
        verbose_s=verbose_s,
        **g_learner_sett
    )

    # ToDo
    plt.figure()
    plt.plot(graph.w_mat[graph.adj_mat == 1], graph_learner.s_mat[graph.adj_mat == 1], 'k*')

    # ----- Save to file -----
    if do_save:
        save_dic = g_learner_sett.copy()
        save_dic.update(ext_graph_sett)
        save_dic.update(dataset_sett)
        graph_learner.save_to_file(
            savepath=save_path,
            filename='graphlearner' + Logger.stringify(save_dic),
        )

    # ----- Plotting -----
    plt.figure()

    mask_tr = ~np.isnan(rating_mat_tr)
    mask_te = ~np.isnan(rating_mat_te)

    rating_mat_pr = graph_learner.x_mat[:n_user]

    plt.subplot(2, 2, 1)
    plt.plot(rating_mat_tr[mask_tr], rating_mat_pr[mask_tr], 'k*')
    plt.title('x (train)')

    plt.subplot(2, 2, 2)
    plt.plot(rating_mat_te[mask_te], rating_mat_pr[mask_te], 'k*')
    plt.title('x (test)')

    rating_mat_pr = graph_learner.predict(graph_learner.x_mat)

    plt.subplot(2, 2, 3)
    plt.plot(rating_mat_tr[mask_tr], rating_mat_pr[mask_tr], 'k*')
    plt.title('S times x (train)')

    plt.subplot(2, 2, 4)
    plt.plot(rating_mat_te[mask_te], rating_mat_pr[mask_te], 'k*')
    plt.title('S times x (test)')
