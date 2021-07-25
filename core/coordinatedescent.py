

from app.models.graphlearning import GraphLearner
from app.utils.log import Logger
from app.utils.mathtools import rmse


class GraphLearningCD:
    def __init__(self, g_learner: GraphLearner, logger: Logger):
        self.g_learner = g_learner
        self.logger = logger

    def run(self, x_0_mat, rat_mat_tr, rat_mat_va, rat_mat_te, n_iter, **kwargs):
        self.g_learner.x_mat = x_0_mat

        # Initial evaluations
        rat_mat_pr = self.g_learner.x_mat[:-1]

        rmse_tr = rmse(rat_mat_tr, rat_mat_pr)
        rmse_va = rmse(rat_mat_va, rat_mat_pr)
        rmse_te = rmse(rat_mat_te, rat_mat_pr)
        self.logger.log(rmse_tr, rmse_va, rmse_te)

        rat_mat_pr = self.g_learner.predict(self.g_learner.x_mat)

        rmse_tr = rmse(rat_mat_tr, rat_mat_pr)
        rmse_va = rmse(rat_mat_va, rat_mat_pr)
        rmse_te = rmse(rat_mat_te, rat_mat_pr)
        self.logger.log(rmse_tr, rmse_va, rmse_te)

        for it in range(n_iter):
            verbose = kwargs['verbose']

            # Update x_mat
            if verbose:
                print('Updating x_mat ...')
            self.g_learner.fit_x(min_val=kwargs['min_val'],
                                 max_val=kwargs['max_val'],
                                 max_distance_to_rated=kwargs['max_distance_to_rated'],
                                 l2_lambda=kwargs['l2_lambda_x'],
                                 gamma=kwargs['gamma'])

            rat_mat_pr = self.g_learner.x_mat[:-1]

            rmse_tr = rmse(rat_mat_tr, rat_mat_pr)
            rmse_va = rmse(rat_mat_va, rat_mat_pr)
            rmse_te = rmse(rat_mat_te, rat_mat_pr)
            self.logger.log(rmse_tr, rmse_va, rmse_te)

            # Update s_mat
            if verbose:
                print('Updating s_mat ...')
            self.g_learner.fit_shift_operator(l2_lambda=kwargs['l2_lambda_s'])

            rat_mat_pr = self.g_learner.predict(self.g_learner.x_mat)

            rmse_tr = rmse(rat_mat_tr, rat_mat_pr)
            rmse_va = rmse(rat_mat_va, rat_mat_pr)
            rmse_te = rmse(rat_mat_te, rat_mat_pr)
            self.logger.log(rmse_tr, rmse_va, rmse_te)
