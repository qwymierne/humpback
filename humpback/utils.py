import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def log_likelihood_in_mle(model, X, y, *args, **kwargs):
    model.fit(X, y)
    if hasattr(model, 'loglik_in_mle'):
        ll = model.loglik_in_mle(X, y)
    elif hasattr(model, 'likelihood_in_mle'):
        ll = np.log(model.loglik_in_mle(X, y))
    else:
        raise NotImplementedError(f'{model.__class__.__name__} cannot calculate log likelihood in MLE')

    return ll


def interactions_apply(X, apply_interaction, ic_requirements):
    assert ic_requirements in ['independent', 'required']
    if apply_interaction is None:
        if ic_requirements == 'independent':
            ret = X
        else:
            ret = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)
    elif apply_interaction:
        ret = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)
    else:
        if ic_requirements == 'required':
            raise ValueError('For chosen IC interactions must be applied')
        else:
            ret = X
    return ret