import numpy as np
from scipy.linalg import pinv


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


def ls_new(x_mat, y, l2_lambda=0):
    if y.ndim == 1:
        y = y.reshape((-1, 1))

    return pinv(x_mat).dot(y)


def rmse(rat_mat_te, rat_mat_pr, check_to_be_rated=True):
    if check_to_be_rated:
        mask_te = ~np.isnan(rat_mat_te)

        err = rat_mat_te[mask_te] - rat_mat_pr[mask_te]

        assert np.all(~np.isnan(err)), 'at least one rating is predicted as NaN!'

        return np.sqrt(np.mean(err**2))
    else:
        err = rat_mat_te - rat_mat_pr
        return np.sqrt(np.nanmean(err**2))


def list_minus_list(l1, l2):
    return list(set(l1) - set(l2))


def fill_with_row_means(mat_org):
    mat = mat_org.copy()

    for row in range(mat.shape[0]):
        mat[row, np.isnan(mat[row])] = np.nanmean(mat[row])

    return mat
