from abc import ABC, abstractmethod

import numpy as np

from .utils import log_likelihood_in_mle


class InformationCriterion(ABC):
    """
    Information Criterion is function which bases on some model and returns score on some data set according to this
    model. Model behind IC must be capable of calculating log-likelihood in its MLE. Some ICs may base on interactions
    between features (mBIC for instance).

    ----------
    model : object
        model which is capable of running ``fit``, ``predict`` and one of ``loglik_in_mle``, ``likelihood_in_mle``
        methods. It is base for calculating IC value for dataset

    Class attributes
    ----------
    _interactions : {``'independent'``, ``'required'``}, default=``'independent'``
        Information whether interactions between features are needed to calculate value of IC
    """

    _interactions = 'independent'

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class AkaikeInformationCriterion(InformationCriterion):
    """
    Akaike Information Criterion - its value is calculated by formula: `` 2 * log(MLE) - 2 * k`` where ``MLE`` is
    likelihood at Maximum Likelihood Estimator and ``k`` is number of features

    ----------
    model : object
        model which is capable of running ``fit``, ``predict`` and one of ``loglik_in_mle``, ``likelihood_in_mle``
        methods. It is base for calculating IC value for dataset

    Class attributes
    ----------
    _interactions = ``'independent'``
    """
    def __call__(self, X, y, *args, **kwargs):
        """Calculates value of AIC on presented dataset.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to ``X``'s dtype if necessary
        """
        ll, params_num = log_likelihood_in_mle(self.model, X, y, *args, **kwargs)

        return 2 * ll - 2 * params_num


class BayesianInformationCriterion(InformationCriterion):
    """
    Bayesian Information Criterion - its value is calculated by formula: `` 2 * log(MLE) - k * log(n)`` where ``MLE`` is
    likelihood at Maximum Likelihood Estimator and ``k`` is number of features and ``n`` is sample size

    ----------
    model : object
        model which is capable of running ``fit``, ``predict`` and one of ``loglik_in_mle``, ``likelihood_in_mle``
        methods. It is base for calculating IC value for dataset

    Class attributes
    ----------
    _interactions = ``'independent'``
    """

    def __call__(self, X, y, *args, **kwargs):
        """Calculates value of BIC on presented dataset.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to ``X``'s dtype if necessary
        """
        ll, params_num = log_likelihood_in_mle(self.model, X, y, *args, **kwargs)

        n, _ = X.shape
        return 2 * ll - params_num * np.log(n)


class ModifiedBayesianInformationCriterion(InformationCriterion):
    """
    Modified Bayesian Information Criterion is IC designed for columns selection problem with interactions. It assumes
    that interactions are important in dataset. It also try to balance not uniform distribution of subsets sizes by
    parameters ``c1`` and ``c2`` - first one is prior guess of number of important, original columns and second is prior
    guess of number of important interactions. Its value is calculated by formula: `` log(MLE) - 0.5 * (k + q) * log(n)
    - k*log(m/c1  1) - q*log(Ne/c2 - 1)`` where ``MLE`` is likelihood at Maximum Likelihood Estimator, ``k`` is number
    of original features in tested columns subset, ``q`` is number of interactions in tested columns subset,
    ``n`` is sample size, ``m`` is number of columns in whole array and ``Ne`` is number of interactions in whole array,
    ie. ``Ne = m*(m-1)/2``.

    ----------
    model : object
        model which is capable of running ``fit``, ``predict`` and one of ``loglik_in_mle``, ``likelihood_in_mle``
        methods. It is base for calculating IC value for dataset
    p1 : float
        initial guess of proportion of important features in original array
    p2 : float
        initial guess of proportion of important interactions between columns in original array

    Class attributes
    ----------
    _interactions = ``'required'``
    """
    _interactions = 'required'

    def __init__(self, model, p1, p2):
        super().__init__(model)
        self.p1 = p1
        self.p2 = p2

    def __call__(self, X, y, m, k, *args, **kwargs):
        """Calculates value of mBIC on presented dataset, which should be subset of greater
        one.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data. First ``k`` columns should be subset of original features and rest should be subset of interactions
            between original features
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to ``X``'s dtype if necessary
        m : int
            number of columns in original data (original + interactions)
        k : int
            number of chosen original features
        """
        ll, params_num = log_likelihood_in_mle(self.model, X, y, *args, **kwargs)
        n, kq = X.shape
        q = kq - k
        c1 = self.p1 * m
        Ne = m * (m - 1) / 2
        c2 = self.p2 * Ne
        return ll - 0.5 * params_num * np.log(n) - k * np.log(m / c1 - 1) - q * np.log(Ne / c2 - 1)


class ModifiedBayesianInformationCriterion2(InformationCriterion):
    """
    Modified Bayesian Information Criterion 2 is IC designed for columns selection problem and bases also on amount of
    columns in original data its value is calculated by formula: `` 2 * log(MLE) - k*log(n) -2*k*log(m/4) + 2*log(k!)``
    where ``MLE`` is likelihood at Maximum Likelihood Estimator, ``k`` is number of features in tested columns subset,
    ``n`` is sample size and ``m``is number of columns in whole array.

    ----------
    model : object
        model which is capable of running ``fit``, ``predict`` and one of ``loglik_in_mle``, ``likelihood_in_mle``
        methods. It is base for calculating IC value for dataset

    Class attributes
    ----------
    _interactions = ``'independent'``
    """

    def __call__(self, X, y, m, *args, **kwargs):
        """Calculates value of mBIC on presented dataset, which should be subset of greater
        one.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data. It shall be subset of some greater dataset, which we want to evaluate.
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to ``X``'s dtype if necessary
        m : int
            number of columns in original data
        """
        ll, params_num = log_likelihood_in_mle(self.model, X, y, *args, **kwargs)
        n, k = X.shape
        return 2 * ll - params_num * np.log(n) - 2 * k * np.log(m / 4) + 2 * np.sum(np.log(np.arange(1, k + 1)))


AIC = AkaikeInformationCriterion
BIC = BayesianInformationCriterion
mBIC = ModifiedBayesianInformationCriterion
mBIC2 = ModifiedBayesianInformationCriterion2
