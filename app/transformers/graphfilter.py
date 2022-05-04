import numpy as np

from app.transformers.transformer_base import Transformer


class GraphFilter(Transformer):
    def __init__(self, s_mat):
        super().__init__()

        self.s_mat = s_mat

        self.v_mat = None
        self.v_inv_mat = None
        self.s_l_mat = None
        self.s_h_mat = None

    def transform(self, x_mat, **kwargs):
        # Find coefficients of high/low freq. (imaginary part is negligible ToDo: careful analysis is needed)
        x_l_mat = np.real(self.s_l_mat.dot(x_mat))
        x_h_mat = np.real(self.s_h_mat.dot(x_mat))

        return x_l_mat, x_h_mat

    def fit(self, bw=0.5, **kwargs):
        # Eigen-value decomposition
        d, v_mat = np.linalg.eig(self.s_mat)

        # Sort eigen-values and eigen-vectors (ascending)
        indices = np.argsort(np.abs(d))
        d = d[indices]
        v_mat = v_mat[:, indices]
        # ToDo
        v_inv_mat = np.linalg.pinv(v_mat, rcond=3e-4)
        # ToDo
        v_mat = np.linalg.pinv(np.linalg.pinv(v_mat))

        self.v_mat, self.v_inv_mat = v_mat, v_inv_mat

        # Split high/low freq. shift operators
        mid = np.floor(len(d)*(1 - bw)).astype(int)
        self.s_l_mat = v_mat[:, mid:].dot(np.diag(d[mid:])).dot(v_inv_mat[mid:])
        self.s_h_mat = v_mat[:, :mid].dot(np.diag(d[:mid])).dot(v_inv_mat[:mid])

        return
