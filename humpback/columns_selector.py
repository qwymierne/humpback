import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, information_criterion, choice_heuristic, stop_condition):
        self.information_criterion = information_criterion
        self.choice_heuristic = choice_heuristic
        self.stop_condition = stop_condition
        self.chosen_columns_ = None

    def fit(self, X, y):
        # TODO preprocessing
        self.choice_heuristic.fit(X, y)
        subsets_gen = self.choice_heuristic.promising_subsets()
        if self.stop_condition == 'max_all':
            promising_subsets = list(subsets_gen)
            ic_vals = [self.information_criterion(X[:, list(map(bool, s))], y[list(map(bool, s))]) for s in promising_subsets]
            self.chosen_columns_ = promising_subsets[ic_vals.index(max(ic_vals))]
        elif self.stop_condition == 'first_decreasing':
            ic_val = -np.inf
            while True:
                try:
                    subset = next(subsets_gen)
                    new_val = self.information_criterion(X[:, list(map(bool, subset))], y[list(map(bool, subset))])
                    if new_val > ic_val:
                        self.chosen_columns_ = subset
                        ic_val = new_val
                    else:
                        break
                except StopIteration:
                    break
        else:
            raise ValueError(f"stop_condition illegal value: '{self.stop_condition}'")

        return self

    def transform(self, X):
        if self.chosen_columns_ is None:
            raise RuntimeError(f'ColumnsSelector not fitted yet. Run fit or fit_tranform first!')
        return X[:, list(map(bool, self.chosen_columns_))]