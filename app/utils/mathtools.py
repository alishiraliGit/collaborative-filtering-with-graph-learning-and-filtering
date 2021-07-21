import numpy as np


def ls(x_mat, y, l2_lambda=0):
    """
    x_mat * s = y

    :param l2_lambda: l2- regularization
    :param x_mat: [a x b] where a > b
    :param y: [a,] or [a x 1]
    :return s: least square estimation: [b x 1]
    """
    if y.ndim == 1:
        y = y.reshape((-1, 1))

    b = x_mat.shape[1]

    p_inv_x_mat = np.linalg.inv(x_mat.T.dot(x_mat) + l2_lambda*np.eye(b)).dot(x_mat.T)  # [b x a]

    s = p_inv_x_mat.dot(y)

    return s
