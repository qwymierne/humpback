from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
import numpy as np


def loglik_in_mle(self, X, y):
    # assume Gaussian distribution of data
    # shall be run after fit and on same data
    check_is_fitted(self)
    pred_y = self.predict(X)
    pred_y_var = np.var(y)
    residuals = y - pred_y
    rss = np.sum(np.square(residuals))
    # print(X.shape)
    return -(X.shape[0] / 2) * np.log(2 * np.pi * pred_y_var) - rss / (2 * pred_y_var)


# def mle(self):

LinearRegression.loglik_in_mle = loglik_in_mle

