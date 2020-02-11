from __future__ import division, print_function, absolute_import

import numpy as np

from scipy.stats import chi2, multivariate_normal, norm
from scipy.linalg import cholesky

from _multivariate_t_cdf import (_standard_univariate_t_cdf,
                                  _standard_bivariate_t_cdf,
                                  _standard_trivariate_t_cdf,
                                  _standard_multivariate_t_cdf)
from _halton import halton_sample

###############################################################################
# Multivariate Student's t-distribution functions


def _multivariate_t_random(location, scale, dof, n_samples, random_state, method="pseudo"):
    """
    Generate random samples from the multivariate Student's t-distribution

    Parameters
    ----------
    location : array_like, shape (n_features,) or (n_features,1)
        Location parameter of the distribution
        If None, the loc defined in the class initialization is used
    scale : array_like, shape (n_features, n_features)
        Scale matrix of the distribution, must be symmetric and positive definite
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    n_samples : int
        Number of samples to generate, must be a positive integer
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by
        numpy.random.
    method : str, default "pseudo"
        Generate either "pseudo" or "quasi" random
        numbers from the distribution. "Quasi" random
        numbers are drawn using Halton sequence

    Returns
    -------
    X : array_like, shape (num_of_samples, num_of_variables)
        A random sample drawn from the distribution, with rows as samples and columns as variables

    """
    dim = location.shape[0]

    if method == "quasi":
        r = halton_sample(n_samples, dim + 1)
        w = chi2.ppf(r[:, -1], dof) / dof
        z = np.array([norm.ppf(r[np.random.permutation(r.shape[0]), i]) for i in range(dim)]).T
    else:
        w = chi2.rvs(dof, size=n_samples, random_state=random_state) / dof
        z = multivariate_normal.rvs(
            np.zeros(dim), np.eye(dim), n_samples, random_state
        )

    y = z / np.sqrt(w)[:, None] if dim > 1 else (z / np.sqrt(w))[:, np.newaxis]
    return np.dot(y, cholesky(scale)) + location


def _multivariate_t_cdf(x, location, scale, dof, maxpts=1e+7, abseps=1e-6, releps=1e-6):
    """
    Multivariate Student's t cumulative density function.

    Parameters
    ----------
    x : array_like
        Sample, shape (n_samples, n_features)
    location : array_like, shape (n_features,)
        Location parameter of the distribution
    scale : array_like, shape (n_features, n_features)
        Scale matrix of the distribution, must be symmetric and positive definite
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    maxpts: integer
        The maximum number of points to use for integration (used when n_features > 3)
    abseps: float
        Absolute error tolerance (used when n_features > 1)
    releps: float
        Relative error tolerance (used when n_features == 2 or n_features == 3)
    

    Returns
    -------
    cdf : ndarray or scalar
        Cumulative density function evaluated at `x`

    """
    if dof == np.inf:
        return multivariate_normal.cdf(x, mean=location, cov=scale,
                                       maxpts=maxpts, abseps=abseps, releps=releps)

    dim = x.shape[1]
    diag_scale = np.diag(scale) if dim > 1 else scale
    inv_diag_scale = 1 / diag_scale
    sqrt_inv_diag_scale = np.sqrt(inv_diag_scale)
    y = np.dot(x - location, np.diag(sqrt_inv_diag_scale)) if dim > 1 else (x - location) * sqrt_inv_diag_scale
    corr_mat = scale * np.outer(sqrt_inv_diag_scale, sqrt_inv_diag_scale) if dim > 1 else None
    if x.shape[1] == 1:
        f_cdf = _standard_univariate_t_cdf(y, dof)
    elif x.shape[1] == 2:
        f_cdf = _standard_bivariate_t_cdf(y, corr_mat, dof, abseps=abseps, releps=releps)
    elif x.shape[1] == 3:
        f_cdf = _standard_trivariate_t_cdf(y, corr_mat, dof, abseps=abseps, releps=releps)
    else:
        f_cdf = _standard_multivariate_t_cdf(y, corr_mat, dof, tol=abseps, max_evaluations=maxpts)

    f_cdf[f_cdf < 0] = 0
    f_cdf[f_cdf > 1] = 1

    return f_cdf
