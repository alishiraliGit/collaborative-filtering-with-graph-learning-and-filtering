import os
import numpy as np

from app.transformers.graph import Graph
from app.models.graphlearning import CounterfactualGraphLearner, GraphMatrixCompletion, \
    CounterfactualGraphMatrixCompletion
from app.utils.mathtools import fill_with_row_means
from app.utils.log import Logger
from core.ratingsimulator import simulate_six_users_mar
from core.coordinatedescent import GraphLearningCD

if __name__ == '__main__':
    # ----- Settings -----
    simul_sett = {}
    graph_sett = {}
    g_learner_sett = {}

    # General
    n_iter = 5
    do_plot_performance_while_logging = True

    # Path
    save_path = os.path.join('..', 'results', 'demo')
    os.makedirs(save_path, exist_ok=True)

    # Simulation
    simul_sett['n_item'] = 500
    simul_sett['sigma_n'] = 0.1
    simul_sett['p_miss'] = 0.5
    simul_sett['min_val'] = 1
    simul_sett['max_val'] = 5

    # Graph
    graph_sett['min_num_common_items'] = 3
    graph_sett['max_degree'] = 2

    # GLearner
    # g_learner_sett['max_distance_to_rated'] = 1
    # g_learner_sett['l2_lambda_x'] = 0.
    g_learner_sett['gamma'] = 0.
    g_learner_sett['l2_lambda_s'] = 1
    g_learner_sett['beta'] = 1
    g_learner_sett['eps_x'] = 1e-2
    verbose_g_learner = True

    # ----- Simulate data -----
    rating_mat_o, rating_mat_m = simulate_six_users_mar(**simul_sett)
    num_user, num_item = rating_mat_o.shape

    # ----- Init. -----
    # x_mat
    x_0_matrix = fill_with_row_means(np.concatenate((rating_mat_o, np.ones((1, num_item))), axis=0))

    # Logger
    log_sett = g_learner_sett.copy()
    log_sett.update(graph_sett)
    log_sett.update(simul_sett)
    logger_x = Logger(settings=log_sett, save_path=save_path, do_plot=do_plot_performance_while_logging, title='x')
    logger_s = Logger(settings=log_sett, save_path=save_path, do_plot=do_plot_performance_while_logging, title='S')

    # ----- Fit a graph -----
    print('Fitting Graph ...')
    graph = Graph(min_num_common_items=graph_sett['min_num_common_items'],
                  max_degree=graph_sett['max_degree'])
    graph.fit_transform(rating_mat_o)

    # ----- Fit a GLearner -----
    # Init.
    print('Initializing GraphLearner ...')
    g_learner = GraphMatrixCompletion.from_graph_object(graph, ~np.isnan(rating_mat_o))
    g_learner.x_mat = x_0_matrix

    # Fit the ks stats
    # print('Fitting KS-statistics ...')
    # g_learner.fit_ks_statistics(rat_mat=rating_mat_o)

    # ----- CD -----
    # Init.
    cd = GraphLearningCD(g_learner=g_learner,
                         logger_x=logger_x,
                         logger_s=logger_s)

    # Run
    print('Running coordinate descent ...')
    cd.run(x_0_mat=x_0_matrix,
           rat_mat_tr=rating_mat_o,
           rat_mat_va=rating_mat_m,
           rat_mat_te=rating_mat_m,
           n_iter=n_iter,
           verbose=verbose_g_learner,
           min_val=simul_sett['min_val'],
           max_val=simul_sett['max_val'],
           calc_bias=False,
           **g_learner_sett)
