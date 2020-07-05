import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from .utils import interactions_apply


class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, information_criterion, choice_heuristic, stop_condition='max_all', interactions=None,
                 interactions_applied=False):
        self.information_criterion = information_criterion
        self.choice_heuristic = choice_heuristic
        self.stop_condition = stop_condition
        self.interactions = interactions
        self.interactions_applied = interactions_applied

    def fit(self, X, y):
        if self.interactions_applied:
            k = (np.sqrt(8 * X.shape[1] + 1) - 1) / 2
            if not k.is_integer():
                raise ValueError(f'X array shape does not fit to interactions shape - it shall have k*(k +1)/2 columns'
                                 f' for k - number of columns in original data')
            else:
                k = int(k)
            X_i = X
        else:
            k = X.shape[1]
            X_i = interactions_apply(X, self.interactions, self.information_criterion._interactions)

        self.choice_heuristic.fit(X_i, y)
        subsets_gen = self.choice_heuristic
        if self.stop_condition == 'max_all':
            promising_subsets = list(subsets_gen)
            ic_vals = [self.information_criterion(X_i[:, list(map(bool, s))], y, m=X_i.shape[1], k=k)
                       for s in promising_subsets]

            best_columns = promising_subsets[ic_vals.index(max(ic_vals))]

        elif self.stop_condition == 'first_decreasing':
            ic_val = -np.inf
            for subset in subsets_gen:
                new_val = self.information_criterion(X_i[:, list(map(bool, subset))], y, m=X_i.shape[1], k=k)
                if new_val > ic_val:
                    best_columns = subset
                    ic_val = new_val
                else:
                    break
        else:
            raise ValueError(f"stop_condition illegal value: '{self.stop_condition}'")

        self.chosen_columns_ = best_columns if self.interactions else best_columns[:k]

        return self

    def transform(self, X):
        check_is_fitted(self)
        if self.interactions and not self.interactions_applied:
            X = interactions_apply(X, self.interactions, self.information_criterion._interactions)
        return X[:, list(map(bool, self.chosen_columns_))]
