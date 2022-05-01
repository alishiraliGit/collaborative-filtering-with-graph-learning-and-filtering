import numpy as np
from scipy.linalg import pinv


def ls(x_mat, y, l2_lambda=0, weights=np.array([1])):
    """
    x_mat * s = y

    :param l2_lambda: l2-regularization
    :param weights:
    :param x_mat: [a x b] where a > b
    :param y: [a,] or [a x 1]
    :return s: least square estimation: [b x 1]
    """
    a, b = x_mat.shape
    if y.ndim == 1:
        y = y.reshape((-1, 1))

    if len(weights) <= 1:
        weights = np.ones((a,))

    # Apply weights
    sq_w_mat = np.diag(np.sqrt(weights))
    x_mat = sq_w_mat.dot(x_mat)
    y = sq_w_mat.dot(y)

    # Calc. pseudo-inverse
    p_inv_x_mat = np.linalg.inv(x_mat.T.dot(x_mat) + l2_lambda * np.eye(b)).dot(x_mat.T)  # [b x a]

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

        return np.sqrt(np.mean(err ** 2))
    else:
        err = rat_mat_te - rat_mat_pr
        return np.sqrt(np.nanmean(err ** 2))


def list_minus_list(l1, l2):
    return list(set(l1) - set(l2))


def fill_with_row_means(mat_org):
    mat = mat_org.copy()

    for row in range(mat.shape[0]):
        mat[row, np.isnan(mat[row])] = np.nanmean(mat[row])

    # Fill nans with column means
    for col in range(mat.shape[1]):
        mat[np.isnan(mat[:, col]), col] = np.nanmean(mat_org[:, col])

    return mat


def percentile_calculator(mat_org):
    mat = mat_org.copy()

    long_tail = []
    short_tail = []
    numbers = []

    for item in range(mat.shape[1]):
        numbers += [np.sum(~np.isnan(mat[:, item]))]

    quarter_percentile = np.percentile(numbers, 75)
    for item in range(mat.shape[1]):
        num_of_rated_user = np.sum(~np.isnan(mat[:, item]))
        # print(num_of_rated_user)
        if num_of_rated_user > quarter_percentile:
            short_tail += [item]
        else:
            long_tail += [item]

    long_array = np.array(long_tail).reshape(1, -1)
    short_array = np.array(short_tail).reshape(1, -1)
    # idxs = np.concatenate((long_array, short_array), axis=0)

    return long_array, short_array


def ACLT(xmat,test,long, short, num_of_items = 15):
    mat = xmat.copy()
    total_len = 0
    mask = (~np.isnan(test))*1
    xmat = xmat*mask
    for user in range(xmat.shape[0]):
        top_predicted_items = np.argsort(xmat[user, :])[-num_of_items:]
        num = [i for i in list(top_predicted_items) if i in long]
        sub_len = len(num)
        total_len += sub_len
    aclt = total_len/xmat.shape[0]
    precision = rmse(test, xmat)
    return aclt, precision

def bound_within(mat_org, min_val, max_val):
    mat = mat_org.copy()
    mat[mat > max_val] = max_val
    mat[mat < min_val] = min_val

    return mat


def vectorize(mat, order='F'):
    return mat.reshape((-1,), order=order)


def unvectorize(vec, n_row, order='F'):
    return vec.reshape((n_row, -1), order=order)
