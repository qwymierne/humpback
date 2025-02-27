from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
import numpy as np


def loglik_in_mle(self, X, y):
    """Extension to sklearn.linear_model.LinearRegression class. Allows calculating log-likelihood in MLE. Assumes data
    comes from Gaussian Distribution. To make give proper results should run on same as fit method.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to ``X``'s dtype if necessary

        Returns
        -------
        ll : float
            Value of log(MLE) for Linear Regression model
        params_num : int
            Number of parameters in model
        """
    check_is_fitted(self)
    n, k = X.shape
    pred_y = self.predict(X)
    residuals = y - pred_y
    rss = np.sum(np.square(residuals))
    sigma2 = rss / n
    return -(n / 2) * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2), k + 2


LinearRegression.loglik_in_mle = loglik_in_mle

