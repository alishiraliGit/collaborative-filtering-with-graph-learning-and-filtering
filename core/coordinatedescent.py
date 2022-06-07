import numpy as np

from app.models.graphlearning import GraphLearnerBase, GraphMatrixCompletion
from app.utils.log import Logger
from app.utils.mathtools import rmse, ACLT


class GraphLearningCD:
    def __init__(self, g_learner: GraphLearnerBase, logger_x: Logger, logger_s: Logger):
        self.g_learner = g_learner
        self.logger_x = logger_x
        self.logger_s = logger_s

    def run(self, x_0_mat, rat_mat_tr, rat_mat_va, rat_mat_te, n_iter, verbose=False, calc_bias=False, **kwargs):
        self.g_learner.x_mat = x_0_mat

        # Initial evaluations
        rat_mat_pr = self.g_learner.x_mat[:-1]

        rmse_va = rmse(rat_mat_va, rat_mat_pr)
        rmse_te = rmse(rat_mat_te, rat_mat_pr)
        if calc_bias:
            bias_tr = np.nanmean(rat_mat_pr - rat_mat_tr)
            bias_va = np.nanmean(rat_mat_pr - rat_mat_va)
            bias_te = np.nanmean(rat_mat_pr - rat_mat_te)
            self.logger_x.log(np.nan, rmse_va, rmse_te, bias_tr, bias_va, bias_te, log_bias=True)
        else:
            self.logger_x.log(np.nan, rmse_va, rmse_te)

        rat_mat_pr = self.g_learner.predict(self.g_learner.x_mat)

        rmse_tr = rmse(rat_mat_tr, rat_mat_pr)
        rmse_va = rmse(rat_mat_va, rat_mat_pr)
        rmse_te = rmse(rat_mat_te, rat_mat_pr)
        if calc_bias:
            bias_tr = np.nanmean(rat_mat_pr - rat_mat_tr)
            bias_va = np.nanmean(rat_mat_pr - rat_mat_va)
            bias_te = np.nanmean(rat_mat_pr - rat_mat_te)
            self.logger_s.log(rmse_tr, rmse_va, rmse_te, bias_tr, bias_va, bias_te, log_bias=True)
        else:
            self.logger_s.log(rmse_tr, rmse_va, rmse_te)

        for it in range(n_iter):
            # Update s_mat
            if verbose:
                print('Updating s_mat ...')
            self.g_learner.fit_shift_operator(**kwargs)

            rat_mat_pr = self.g_learner.predict(self.g_learner.x_mat)

            rmse_tr = rmse(rat_mat_tr, rat_mat_pr)
            rmse_va = rmse(rat_mat_va, rat_mat_pr)
            rmse_te = rmse(rat_mat_te, rat_mat_pr)
            if calc_bias:
                bias_tr = np.nanmean(rat_mat_pr - rat_mat_tr)
                bias_va = np.nanmean(rat_mat_pr - rat_mat_va)
                bias_te = np.nanmean(rat_mat_pr - rat_mat_te)
                self.logger_s.log(rmse_tr, rmse_va, rmse_te, bias_tr, bias_va, bias_te, log_bias=True)
            else:
                self.logger_s.log(rmse_tr, rmse_va, rmse_te)

            # Update x_mat
            if verbose:
                print('Updating x_mat ...')
            self.g_learner.fit_x(**kwargs)

            rat_mat_pr = self.g_learner.x_mat[:-1]

            rmse_va = rmse(rat_mat_va, rat_mat_pr)
            rmse_te = rmse(rat_mat_te, rat_mat_pr)
            if calc_bias:
                bias_tr = np.nanmean(rat_mat_pr - rat_mat_tr)
                bias_va = np.nanmean(rat_mat_pr - rat_mat_va)
                bias_te = np.nanmean(rat_mat_pr - rat_mat_te)
                self.logger_x.log(np.nan, rmse_va, rmse_te, bias_tr, bias_va, bias_te, log_bias=True)
            else:
                self.logger_x.log(np.nan, rmse_va, rmse_te)

    def k_trial(self, graph: GraphMatrixCompletion, rat_mat_te,  **kwargs):
        results = []
        precisions = []
        k_values = [4, 5, 6.65, 10, 15, 20, 25]
        # k_values = [0.01, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        test_mat = (~np.isnan(rat_mat_te) * 1)
        for k in k_values:
            graph.k_predict(k)
            graph.fit_x(**kwargs)
            result, precision = ACLT(graph.x_mat[:-1], rat_mat_te, graph.long, graph.short)
            results += [result]
            precisions += [precision]
            # graph.fit_shift_coefss(**kwargs)
            # graph.fit_x(**kwargs)
            # result, precision = ACLT(graph.x_mat[:-1], rat_mat_te, graph.long, graph.short)
            # results += [result]
            # precisions += [precision]
        return results, precisions
