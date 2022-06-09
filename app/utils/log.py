from datetime import datetime
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt

from app.utils.mathtools import rmse


def print_red(txt):
    print('\033[91m' + txt + '\033[0m')


class Logger:
    def __init__(self, settings, save_path, title, do_plot=False):
        self.settings = settings
        self.save_path = save_path
        self.do_plot = do_plot
        self.title = title

        self.rmse_tr = []
        self.rmse_va = []
        self.rmse_te = []

        self.bias_tr = []
        self.bias_va = []
        self.bias_te = []

        if do_plot:
            self.fig = plt.figure()
            plt.title(title)
            self.fig_bias = plt.figure()
            plt.title(title)

    def log(self, rmse_tr, rmse_va, rmse_te, bias_tr=np.nan, bias_va=np.nan, bias_te=np.nan, log_bias=False):
        self.rmse_tr += [rmse_tr]
        self.rmse_va += [rmse_va]
        self.rmse_te += [rmse_te]

        print('iteration: %d, rmse train: %.3f, rmse val: %.3f rmse test: %.3f' %
              (len(self.rmse_tr), self.rmse_tr[-1], self.rmse_va[-1], self.rmse_te[-1]))

        if log_bias:
            self.bias_tr += [bias_tr]
            self.bias_va += [bias_va]
            self.bias_te += [bias_te]

            print('bias train: %.3f, bias val: %.3f bias test: %.3f' %
                  (self.bias_tr[-1], self.bias_va[-1], self.bias_te[-1]))

        if self.do_plot:
            plt.figure(self.fig.number)

            plt.plot(len(self.rmse_tr), self.rmse_tr[-1], 'ro')
            plt.plot(len(self.rmse_va), self.rmse_va[-1], 'ko')
            plt.plot(len(self.rmse_te), self.rmse_te[-1], 'bo')
            plt.legend(['train', 'validation', 'test'])
            plt.ylabel('rmse')
            plt.xlabel('#iter')

            if log_bias:
                plt.figure(self.fig_bias.number)

                plt.plot(len(self.bias_tr), self.bias_tr[-1], 'ro')
                plt.plot(len(self.bias_va), self.bias_va[-1], 'ko')
                plt.plot(len(self.bias_te), self.bias_te[-1], 'bo')
                plt.legend(['train', 'validation', 'test'])
                plt.ylabel('average bias')
                plt.xlabel('#iter')

            plt.pause(0.05)

    def eval_and_log(self, rat_mat_pr, rat_mat_tr, rat_mat_va, rat_mat_te, calc_bias=False):
        rmse_tr = rmse(rat_mat_tr, rat_mat_pr)
        rmse_va = rmse(rat_mat_va, rat_mat_pr)
        rmse_te = rmse(rat_mat_te, rat_mat_pr)

        if calc_bias:
            bias_tr = np.nanmean(rat_mat_pr - rat_mat_tr)
            bias_va = np.nanmean(rat_mat_pr - rat_mat_va)
            bias_te = np.nanmean(rat_mat_pr - rat_mat_te)
            self.log(rmse_tr, rmse_va, rmse_te, bias_tr, bias_va, bias_te, log_bias=True)
        else:
            self.log(rmse_tr, rmse_va, rmse_te)


    def save(self, ext=None):
        stringified = 'result' + Logger.stringify(self.settings)

        file_name = stringified + '-' + datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        with open(os.path.join(self.save_path, file_name + '.res'), 'wb') as f:
            pickle.dump({'rmse_tr': self.rmse_tr,
                         'rmse_va': self.rmse_va,
                         'rmse_te': self.rmse_te,
                         'settings': self.settings}, f)

        if ext is not None:
            with open(os.path.join(self.save_path, file_name + '.extres'), 'wb') as f:
                pickle.dump(ext, f)

    @staticmethod
    def load(load_path, file_name, load_ext=False):
        file_path = os.path.join(load_path, file_name)
        if load_ext:
            file_path += '.extres'
        else:
            file_path += '.res'

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def stringify(dic):
        stringified = ''
        for key, val in dic.items():
            stringified += '-'
            stringified += key
            if isinstance(val, int):
                stringified += str(val)
            elif isinstance(val, float):
                stringified += '%.0e' % val
            elif isinstance(val, str):
                stringified += val
            else:
                stringified += Logger.stringify(val)

        return stringified
