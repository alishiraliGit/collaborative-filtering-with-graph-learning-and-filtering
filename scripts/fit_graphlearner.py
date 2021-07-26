import os
import numpy as np
from matplotlib import pyplot as plt

from app.transformers.graph import Graph
from app.models.graphlearning import GraphLearner
from app.utils.mathtools import fill_with_row_means
from app.utils.log import Logger
from app.utils.data_handler import load_dataset
from core.coordinatedescent import GraphLearningCD


if __name__ == '__main__':
    # ----- Settings -----
    dataset_sett = {}
    graph_sett = {}
    g_learner_sett = {}

    # General
    n_iter = 1
    do_plot_performance_while_logging = False

    # Path
    data_load_path = os.path.join('..', 'data', 'ml-100k')
    graph_load_path = os.path.join('..', 'results', 'graphs')

    save_path = os.path.join('..', 'results', 'graphlearners')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_sett['dataset_name'] = 'ml-100k'
    dataset_sett['part'] = 4
    dataset_sett['do_transpose'] = False

    # Graph
    graph_sett['min_num_common_items'] = 10
    graph_sett['max_degree'] = 3

    # GraphLearner
    g_learner_sett['max_distance_to_rated'] = 1
    g_learner_sett['l2_lambda_x'] = 0
    g_learner_sett['gamma'] = 1
    g_learner_sett['l2_lambda_s'] = 10
    verbose_g_learner = True

    # ----- Load graph -----
    print('Loading graph ...')
    graph_load_sett = graph_sett.copy()
    graph_load_sett.update(dataset_sett)
    graph, graph_dic = Graph.load_from_file(load_path=graph_load_path,
                                            file_name='graph' + Logger.stringify(graph_load_sett))

    # ------- Load data -------
    print('Loading data ...')
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item = \
        load_dataset(data_load_path, **graph_dic['ext']['dataset'])

    # ----- Init. -----
    print('Initializing ...')
    # x_mat
    x_0_matrix = fill_with_row_means(np.concatenate((rating_mat_tr, np.ones((1, n_item))), axis=0))

    # Graph learner
    user_item_is_rated_mat = ~np.isnan(rating_mat_tr)
    graph_learner = GraphLearner.from_graph_object(graph, user_item_is_rated_mat)

    # Logger
    log_sett = g_learner_sett.copy()
    log_sett.update(graph_sett)
    log_sett.update(dataset_sett)
    logger = Logger(settings=log_sett, save_path=save_path, do_plot=do_plot_performance_while_logging)

    # ----- Coordinate descent -----
    cd = GraphLearningCD(g_learner=graph_learner,
                         logger=logger)

    cd.run(x_0_mat=x_0_matrix,
           rat_mat_tr=rating_mat_tr,
           rat_mat_va=rating_mat_va,
           rat_mat_te=rating_mat_te,
           n_iter=n_iter,
           verbose=verbose_g_learner,
           min_val=graph_dic['ext']['dataset']['min_value'],
           max_val=graph_dic['ext']['dataset']['max_value'],
           **g_learner_sett)

    # ----- Save to file -----
    save_dic = g_learner_sett.copy()
    save_dic.update(graph_sett)
    save_dic.update(dataset_sett)
    graph_learner.save_to_file(savepath=save_path,
                               file_name='graphlearner' + Logger.stringify(save_dic),
                               ext_dic={'dataset': dataset_sett})

    # ----- Plotting -----
    plt.figure()

    mask_tr = ~np.isnan(rating_mat_tr)
    mask_te = ~np.isnan(rating_mat_te)

    rating_mat_pr = graph_learner.x_mat[:-1]

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
