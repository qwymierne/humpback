from abc import ABC, abstractmethod

import numpy as np

from .utils import log_likelihood_in_mle


class InformationCriterion(ABC):

    _interactions = 'independent'

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class AkaikeInformationCriterion(InformationCriterion):

    def __call__(self, X, y, *args, **kwargs):
        ll = log_likelihood_in_mle(self.model, X, y, *args, **kwargs)

        return 2 * ll - 2 * X.shape[1]


class BayesianInformationCriterion(InformationCriterion):

    def __call__(self, X, y, *args, **kwargs):
        ll = log_likelihood_in_mle(self.model, X, y, *args, **kwargs)

        n, k = X.shape
        return 2 * ll - k * np.log(n)


class ModifiedBayesianInformationCriterion(InformationCriterion):

    _interactions = 'required'

    def __init__(self, model, c1, c2):
        super().__init__(model)
        self.c1 = c1
        self.c2 = c2

    def __call__(self, X, y, m, k, *args, **kwargs):
        ll = log_likelihood_in_mle(self.model, X, y, *args, **kwargs)
        n, kq = X.shape
        q = kq - k
        return ll - 0.5 * (k + q) * np.log(n) - k * np.log(m / self.c1 - 1) - q * np.log(m * (m - 1) / (2 * self.c2) - 1)


class ModifiedBayesianInformationCriterion2(InformationCriterion):

    def __call__(self, X, y, m, *args, **kwargs):
        ll = log_likelihood_in_mle(self.model, X, y, *args, **kwargs)
        n, k = X.shape
        return 2 * ll - k * np.log(n) - 2 * k * np.log(m / 4) + 2 * np.sum(np.log(np.arange(1, k + 1)))


AIC = AkaikeInformationCriterion
BIC = BayesianInformationCriterion
mBIC = ModifiedBayesianInformationCriterion
mBIC2 = ModifiedBayesianInformationCriterion2
