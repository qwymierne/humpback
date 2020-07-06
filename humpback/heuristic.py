from sklearn.linear_model import lasso_path
from sklearn.utils.validation import check_is_fitted
import numpy as np


class LassoPathHeuristic:
    """Applies lasso path algorithm to data. Its interface is consistent with ColumnsSelector ie. it is iterator. Each
    element in iteration is subset of columns which activates and some level of parameter alpha, so each subset of
    columns is greater then previous one (and in most cases it involves one more element).

    Parameters
    ----------
    eps : float, default=5e-3
        Length of the path. ``eps=5e-3`` means that
        ``alpha_min / alpha_max = 5e-3``
    fit_intercept : bool, default=True
        Whether to fit an intercept or not
    Attributes
    ----------
    coefs_ :  ndarray of shape (n_features, n_alphas) or (n_outputs, n_features, n_alphas)
        Coefficients obtained from lasso_path in unchanged version. Basing on them, during iteration new subsets are
        generated.
    """
    def __init__(self, eps=5e-3, fit_intercept=True):
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
        """Runs lasso path algorithm to obtain coefficients

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data to avoid
            unnecessary memory duplication. If ``y`` is mono-output then ``X``
            can be sparse.
        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)
            Target values
        """
        _, coefs, _ = lasso_path(X, y, eps=self.eps, fit_intercept=self.fit_intercept)
        self.coefs_ = coefs

        return self
