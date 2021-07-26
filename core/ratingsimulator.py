import numpy as np
from numpy.random import default_rng

from app.utils.mathtools import bound_within


def simulate_six_users_mar(n_item, sigma_n, p_miss, min_val=1, max_val=5):
    rng = default_rng(0)

    # Simulate raw ratings
    r1 = rng.integers(low=min_val, high=max_val, size=(1, n_item)).astype(float)
    rat_mat_1 = np.concatenate((r1, 0.8*r1, 0.5*r1 + 1), axis=0)

    r2 = rng.integers(low=min_val, high=max_val, size=(1, n_item)).astype(float)
    rat_mat_2 = np.concatenate((r2, 0.7*r2, 0.5*r2 + 2), axis=0)

    rat_mat_all = np.concatenate((rat_mat_1, rat_mat_2), axis=0)
    rat_mat_all += rng.normal(loc=0, scale=sigma_n, size=rat_mat_all.shape)
    rat_mat_all = bound_within(rat_mat_all, min_val, max_val)
    rat_mat_all = np.round(rat_mat_all)

    # Remove random elements
    mask_nan = rng.random(size=rat_mat_all.shape) < p_miss

    rat_mat_o = rat_mat_all.copy()
    rat_mat_o[mask_nan] = np.nan

    rat_mat_m = rat_mat_all.copy()
    rat_mat_m[~mask_nan] = np.nan

    return rat_mat_o, rat_mat_m
