import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import interactions_apply


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """
        Columns Selector transformer basing on Information Criterion. Chooses best columns subset
        from proposed by Heuristic. Also supports interactions between columns ie. pairwise
        products and cooperate with IC needs them (like mBIC).

        ----------
        information_criterion : InformationCriterion
            IC used to evaluate subset of columns, must be callable object and work in maximum manner i.e. the greater
            value the better subset
        choice_heuristic : heuristic
            Heuristic used to generate subsets of columns. Heuristic shall be an iterator which every time next is call
            return new guess. Subsets of columns are encoded as binary mask i.e. 1's for chosen columns and 0's for rest
        stop_condition : {``'max_all'``, ``'first_decreasing'``}, default=``'max_all'``,
            informs when heuristic's proposals iteration shall stop - if set to ``'max_all'`` all proposal will be
            evaluate and if set to ``'first_decreasing'`` iteration will stop when IC value of some proposition is lower
            than IC value of previous one and that previous one is chosen
        interactions : bool or None, default=None,
            informs whether interactions between columns shall be involved into consideration - if True then subset of
            columns will be extended at the beginning of process and also some interactions might be added to
            transformed array. If set ot None then default behaviour of IC would be used (for example mBIC needs
            interactions to evaluate - see at IC's ``'_interactions'`` class attribute), but columns will always be
            subset of original ones. If False interactions won't be applied, but it may cause Error in situation where
            IC needs interactions and they are not created earlier (see ``interactions_applied``)
        interactions_applied : bool, default=False,
            informs if interactions where applied earlier to data

        Attributes
        ----------
        chosen_columns_ : binary array of shape (n_features) or (1, n_features)
            Subset of columns which maximize IC on data passed in fit method
        """

    def __init__(self, information_criterion, choice_heuristic, stop_condition='max_all', interactions=None,
                 interactions_applied=False):
        self.information_criterion = information_criterion
        self.choice_heuristic = choice_heuristic
        self.stop_condition = stop_condition
        self.interactions = interactions
        self.interactions_applied = interactions_applied

    def fit(self, X, y):
        """Fit transformer to choose best columns subset.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to ``X``'s dtype if necessary
        """

        if self.interactions_applied:
            m = (np.sqrt(8 * X.shape[1] + 1) - 1) / 2
            if not m.is_integer():
                raise ValueError(f'X array shape does not fit to interactions shape - it shall have k*(k + 1)/2 columns'
                                 f' for k - number of columns in original data')
            else:
                m = int(m)
            X_i = X
        else:
            X_i, m = interactions_apply(X, self.interactions, self.information_criterion._interactions)
        self.choice_heuristic.fit(X_i, y)
        subsets_gen = self.choice_heuristic
        if self.stop_condition == 'max_all':
            promising_subsets = list(subsets_gen)
            ic_vals = [self.information_criterion(X_i[:, list(map(bool, s))], y, m=m,  k=sum(s[:m]))
                       for s in promising_subsets]

            best_columns = promising_subsets[ic_vals.index(max(ic_vals))]

        elif self.stop_condition == 'first_decreasing':
            ic_val = -np.inf
            for subset in subsets_gen:
                new_val = self.information_criterion(X_i[:, list(map(bool, subset))], y, m=m, k=sum(subset[:m]))
                if new_val > ic_val:
                    best_columns = subset
                    ic_val = new_val
                else:
                    break
        else:
            raise ValueError(f"stop_condition illegal value: '{self.stop_condition}'")

        self.chosen_columns_ = best_columns if self.interactions else best_columns[:m]

        return self

    def transform(self, X):
        """ Removes from data columns, which are considered as not important ones
               Parameters
               ----------
               X : array-like, shape [n_samples, n_features]
                   Data to filter.
               """
        check_is_fitted(self)
        if self.interactions and not self.interactions_applied:
            X = interactions_apply(X, self.interactions, self.information_criterion._interactions)
        return X[:, list(map(bool, self.chosen_columns_))]
