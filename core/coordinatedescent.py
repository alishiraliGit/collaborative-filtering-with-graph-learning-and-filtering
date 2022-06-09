import numpy as np

from app.models.graphlearning import GraphLearnerBase
from app.utils.log import Logger, print_red
from app.utils.mathtools import rmse


class GraphLearningCD:
    def __init__(self, g_learner: GraphLearnerBase, logger_x: Logger, logger_s: Logger):
        self.g_learner = g_learner
        self.logger_x = logger_x
        self.logger_s = logger_s

    def run(self, x_0_mat, rat_mat_tr, rat_mat_va, rat_mat_te, n_iter, verbose=False, calc_bias=False, **kwargs):
        n_user = rat_mat_tr.shape[0]

        self.g_learner.x_mat = x_0_mat

        # Initial evaluations
        rat_mat_pr = self.g_learner.x_mat[:n_user]
        self.logger_x.eval_and_log(rat_mat_pr, rat_mat_tr, rat_mat_va, rat_mat_te, calc_bias)

        rat_mat_pr = self.g_learner.predict(self.g_learner.x_mat)
        self.logger_s.eval_and_log(rat_mat_pr, rat_mat_tr, rat_mat_va, rat_mat_te, calc_bias)

        # ToDo
        # self.g_learner.fit_scale_of_shift_operator()
        # rat_mat_pr = self.g_learner.predict(self.g_learner.x_mat)
        # self.logger_s.eval_and_log(rat_mat_pr, rat_mat_tr, rat_mat_va, rat_mat_te, calc_bias)

        for it in range(n_iter):
            print('=====================')
            # Update s_mat
            if verbose:
                print('Updating s_mat ...')
            self.g_learner.fit_shift_operator(**kwargs)

            rat_mat_pr = self.g_learner.predict(self.g_learner.x_mat)
            self.logger_s.eval_and_log(rat_mat_pr, rat_mat_tr, rat_mat_va, rat_mat_te, calc_bias)

            # Update x_mat
            if verbose:
                print('Updating x_mat ...')
            self.g_learner.fit_x(**kwargs)

            rat_mat_pr = self.g_learner.x_mat[:n_user]
            self.logger_x.eval_and_log(rat_mat_pr, rat_mat_tr, rat_mat_va, rat_mat_te, calc_bias)
