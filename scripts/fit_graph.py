import os

from app.transformers.graph import Graph, SymmetricGraph, SmoothGraph
from app.utils.data_handler import load_dataset
from app.utils.log import Logger


if __name__ == '__main__':
    # ----- Settings -----
    dataset_sett = {}
    graph_sett = {}

    # Path
    load_path = os.path.join('..', 'data', 'ml-100k')

    save_path = os.path.join('..', 'results', 'graphs')
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    dataset_sett['name'] = 'ml-100k'
    dataset_sett['part'] = 3
    dataset_sett['min_value'] = 1
    dataset_sett['max_value'] = 5

    # Cross-validation
    dataset_sett['va_split'] = 0.1
    dataset_sett['random_state'] = 1

    # Item-based (True) or user-based
    dataset_sett['do_transpose'] = False

    # Graph
    graph_sett['min_num_common_items'] = 8
    graph_sett['max_degree'] = 5
    graph_sett['min_degree'] = 3  # For symmetric and smooth graphs only

    assert graph_sett['max_degree'] <= graph_sett['min_num_common_items']

    # Save
    save_dic = graph_sett.copy()
    save_dic.update({'dataset_name': dataset_sett['name'],
                     'part': dataset_sett['part'],
                     'do_transpose': dataset_sett['do_transpose']})

    # ------- Load data -------
    print('Loading data ...')
    rating_mat_tr, rating_mat_va, rating_mat_te, n_user, n_item = \
        load_dataset(load_path, **dataset_sett)

    # ----- Fit graph structure -----
    print('Fitting graph on ratings ...')
    graph = SmoothGraph(
        min_num_common_items=graph_sett['min_num_common_items'],
        max_degree=graph_sett['max_degree'],
        min_degree=graph_sett['min_degree']  # For symmetric and smooth graphs only
    )

    graph.fit_transform(rating_mat_tr)

    # ----- Save to file -----
    print('Saving graph to file ...')
    graph.save_to_file(
        save_path=save_path,
        file_name='graph' + Logger.stringify(save_dic),
        ext_dic={'dataset': dataset_sett}
    )
