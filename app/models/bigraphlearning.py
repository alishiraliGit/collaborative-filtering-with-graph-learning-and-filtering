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
from app.models.graphlearning import GraphLearner
from app.transformers.graph import Graph
from app.utils.mathtools import ls, list_minus_list, vectorize, unvectorize


class BiGraphLearnerBase(Model, abc.ABC):
    def __init__(self, adj_mat_v, adj_mat_h, ui_is_rated_mat):
        self._adj_mat_v = adj_mat_v  # [n_user x n_user]
        self._adj_mat_h = adj_mat_h  # [n_item x n_item]
        self._ui_is_rated_mat = ui_is_rated_mat  # [n_user x n_item]
        self._n_user, self._n_item = ui_is_rated_mat.shape

        self.x_mat = None  # [n_user x n_item]

    @staticmethod
    def from_graph_object(g_v: Graph, g_h: Graph, ui_is_rated_mat):
        pass

    @abc.abstractmethod
    def fit_shift_operators(self, **kwargs):
        pass

    @abc.abstractmethod
    def fit_x(self, **kwargs):
        pass


class BiGraphMatrixCompletion(BiGraphLearnerBase):
    def __init__(self, adj_mat_v, adj_mat_u, ui_is_rated_mat):
        super().__init__(adj_mat_v, adj_mat_u, ui_is_rated_mat)

        self.s_mat = None  # [(n_user + 1) x (n_user + 1)]
        self.q_mat = None  # [(n_item + 1) x (n_item + 1)]

    @staticmethod
    def from_graph_object(g_v: Graph, g_h: Graph, ui_is_rated_mat):
        bgmc = BiGraphMatrixCompletion(g_v.adj_mat, g_h.adj_mat, ui_is_rated_mat)

        g_learner_v = GraphLearner.from_graph_object(g_v, ui_is_rated_mat)
        g_learner_h = GraphLearner.from_graph_object(g_h, ui_is_rated_mat.T)

        bgmc.s_mat = g_learner_v.s_mat
        bgmc.q_mat = g_learner_h.s_mat.T

        return bgmc

    @abc.abstractmethod
    def fit_shift_operators(self, **kwargs):
        pass

    @abc.abstractmethod
    def fit_x(self, **kwargs):
        pass

    def predict(self, x_mat, **kwargs):
        # ToDo
        pass

    def save_to_file(self, savepath, file_name, ext_dic=None):
        # ToDo
        pass

    @staticmethod
    def load_from_file(loadpath, file_name):
        # ToDo
        pass