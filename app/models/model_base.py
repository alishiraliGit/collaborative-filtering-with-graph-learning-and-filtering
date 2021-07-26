import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, data_tr, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, data_te, **kwargs):
        pass

    @abc.abstractmethod
    def save_to_file(self, savepath, file_name, ext_dic=None):
        pass

    @staticmethod
    def load_from_file(loadpath, file_name):
        pass
