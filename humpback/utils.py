import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def log_likelihood_in_mle(model, X, y, *args, **kwargs):
    """Try to calculate log-likelihood in mle on some model. If model is not supporting it raise Error.
        Parameters
        ----------
        model : object
            model which log-likelihood in MLE is calculated
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to ``X``'s dtype if necessary

        Returns
        -------
        ll : float
            Value of log(MLE) for Linear Regression model
        """
    model.fit(X, y)
    if hasattr(model, 'loglik_in_mle'):
        ll = model.loglik_in_mle(X, y)
    elif hasattr(model, 'likelihood_in_mle'):
        ll = np.log(model.likelihood_in_mle(X, y))
    else:
        raise NotImplementedError(f'{model.__class__.__name__} cannot calculate log likelihood in MLE')

    return ll


def interactions_apply(X, apply_interaction, ic_requirements):
    """Calculates or no interactions between data and eventually concat them to data.
    Parameters
    ----------
    X : {ndarray, sparse matrix} of (n_samples, n_features)
        Data
    apply_interaction : bool or None
        Information whether interactions shall be applied or not. If None interactions are applied according to value of
        ``ic_requirements``. If False and ``ic_requirements``= ``'required'`` error is raised
    ic_requirements : {'independent', 'required'}
        Determines behaviour if ``apply_interactions``=``None``. Also says what number should be considered as original
        data size.
    Returns
    -------
    ret : {ndarray, sparse matrix} of (n_samples, n_features)
        Data with possibly
    m : int
        Size of original data, determined by ``ic_requirements``
    """
    assert ic_requirements in ['independent', 'required']
    if apply_interaction is None:
        m = X.shape[1]
        if ic_requirements == 'independent':
            ret = X
        else:
            ret = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)
    elif apply_interaction:
        m = X.shape[1] if ic_requirements == 'required' else X.shape[1] * (X.shape[1] + 1) // 2
        ret = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)
    else:
        if ic_requirements == 'required':
            raise ValueError('For chosen IC interactions must be applied')
        else:
            ret = X
            m = X.shape[1]
    return ret, m
