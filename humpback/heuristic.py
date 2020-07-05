from sklearn.linear_model import lasso_path
from sklearn.utils.validation import check_is_fitted
import numpy as np


class LassoPathHeuristic:

    def __init__(self, eps=5e-3, fit_intercept=False):
        self.eps = eps
        self.fit_intercept = fit_intercept

    def __iter__(self):
        check_is_fitted(self)
        prev_params = np.zeros((self.coefs_.shape[0], 1))
        for c in np.where(self.coefs_ != 0., 1., 0.).T:
            if (c != prev_params).any():
                prev_params = c
                yield c

    def fit(self, X, y):
        _, coefs, _ = lasso_path(X, y, eps=self.eps, fit_intercept=self.fit_intercept)
        self.coefs_ = coefs

        return self
