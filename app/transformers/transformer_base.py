import abc


class Transformer(abc.ABC):
    @abc.abstractmethod
    def fit(self, data_tr, **kwargs):
        pass

    @abc.abstractmethod
    def transform(self, data_te, **kwargs):
        pass

    def fit_transform(self, data, **kwargs):
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
