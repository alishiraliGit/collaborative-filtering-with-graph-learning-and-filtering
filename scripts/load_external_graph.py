import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

from app.transformers.graph import SmoothGraph
from app.utils.log import Logger


if __name__ == '__main__':
    # ----- Settings -----
    dataset_sett = {}
    graph_sett = {}

    # General
    do_save = True

    # Path
    ext_graph_load_path = os.path.join('..', 'results', 'graphs', 'ext')
    ext_graph_filename = 'graph_min8_a0_b-3'

    graph_load_path = os.path.join('..', 'results', 'graphs')

    save_path = os.path.join('..', 'results', 'graphs', 'ext')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_sett['dataset_name'] = 'ml-100k'
    dataset_sett['part'] = 3
    dataset_sett['do_transpose'] = False

    # Graph
    graph_sett['min_num_common_items'] = 6
    graph_sett['max_degree'] = 5  # 5
    graph_sett['min_degree'] = 1  # Only for symmetric graphs, 1

    # ----- Load ext graph -----
    ext_graph_dic = loadmat(os.path.join(ext_graph_load_path, ext_graph_filename))
    ext_graph_w_mat = ext_graph_dic['ww']

    # ----- Load graph -----
    print('Loading graph ...')
    graph_load_sett = graph_sett.copy()
    graph_load_sett.update(dataset_sett)
    graph, graph_dic = SmoothGraph.load_from_file(
        load_path=graph_load_path,
        file_name='graph' + Logger.stringify(graph_load_sett) + '_nopruning'
    )

    # ----- Initial comparison -----
    print('corr coef of w matrices:')
    print(np.corrcoef(ext_graph_w_mat.reshape((-1,)), graph.w_mat.reshape((-1,))))

    plt.figure()
    plt.plot(ext_graph_w_mat.reshape((-1,)), graph.w_mat.reshape((-1,)), 'k*')
    plt.xlabel('weights of the ext graph')
    plt.ylabel('weights of the baseline graph')

    # ----- Clean ext graph -----
    plt.figure()
    plt.hist(ext_graph_w_mat.reshape((-1,)), bins=100, color='k')

    ext_th = 0

    ext_graph_w_cleaned_mat = ext_graph_w_mat.copy()
    ext_graph_w_cleaned_mat[ext_graph_w_cleaned_mat < ext_th] = 0

    # ----- Save cleaned ext graph -----
    # savemat(
    #     os.path.join('..', 'ext', 'graphsmat', 'graph' + Logger.stringify(graph_load_sett) + '_nopruning.mat')
    #     , {'adj_mat': graph.adj_mat, 'w_mat': graph.w_mat}
    # )

    if do_save:
        with open(os.path.join(save_path, ext_graph_filename + '.pkl'), 'wb') as f:
            pickle.dump({
                'w_mat': ext_graph_w_cleaned_mat,
                'adj_mat': (ext_graph_w_cleaned_mat > 0)*1
            }, f)
