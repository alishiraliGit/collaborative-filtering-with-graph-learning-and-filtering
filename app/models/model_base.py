import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, data_tr, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, data_te, **kwargs):
        pass
