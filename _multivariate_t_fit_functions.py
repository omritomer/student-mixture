"""Student's t-distribution Fitting methods"""

# Author: Omri Tomer <omritomer1@mail.tau.ac.il>
# License: BSD 3 clause

import numpy as np

from scipy import linalg
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import newton
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_array

import warnings

###############################################################################
# Functions to be used by the MultivariateTFit class


def _check_X(X, n_features=None, ensure_min_samples=1):
    """Check the input data X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)

    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


def _check_shape(param, param_shape, name):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : string
    """
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError("The parameter '%s' should have the shape of %s, "
                         "but got %s" % (name, param_shape, param.shape))


def _check_location(location, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    mean : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    location = check_array(location, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(location, (n_features,), 'location')
    return location


def _check_dof(dof):
    """Check the user provided 'dofs'.

    Parameters
    ----------
    dofs : array-like, shape (n_components,)
        The degrees-of-freedom of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    # check range
    if dof < 2:
        raise ValueError("The parameter 'dof' should be in the range "
                         "[2, inf), but got min value %.5f"
                         % dof)
    return dof


def _check_precision_positivity(precision, scale_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % scale_type)


def _check_precision_matrix(precision, scale_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(linalg.eigvalsh(precision) > 0.)):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % scale_type)


def _check_precision_full(precision, scale_type):
    """Check the precision matrices are symmetric and positive-definite."""
    _check_precision_matrix(precision, scale_type)


def _check_precision(precision, scale_type, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_features, n_features)
        'diag' : shape of (n_features,)
        'spherical' : scalar

    scale_type : string

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precision = check_array(precision, dtype=[np.float64, np.float32],
                            ensure_2d=False,
                            allow_nd=False)

    precision_shape = {'full': (n_features, n_features),
                       'diag': (n_features,),
                       'spherical': (1,)}
    _check_shape(precision, precision_shape[scale_type],
                 '%s precision' % scale_type)

    _check_precision = {'full': _check_precision_matrix,
                        'diag': _check_precision_positivity,
                        'spherical': _check_precision_positivity}
    _check_precision[scale_type](precision, scale_type)
    return precision


###############################################################################
# Student mixture parameters estimators (used by the M-Step)

def _estimate_student_scales_full(gamma_priors, X, location, reg_scale):
    """Estimate the full scale matrix.

    Parameters
    ----------
    gamma_priors : array-like, shape (n_samples,)

    X : array-like, shape (n_samples, n_features)

    location : array-like, shape (n_features,)

    reg_scale : float

    Returns
    -------
    scales : array, shape (n_features, n_features)
        The scale matrix of the distribution.
    """
    n_samples, _ = X.shape
    diff = X - location
    scale = np.dot((gamma_priors[:, np.newaxis] * diff).T, diff) / n_samples
    return scale + reg_scale


def _estimate_student_scales_diag(gamma_priors, X, location, reg_scale):
    """Estimate the diagonal scale vector.

    Parameters
    ----------
    gamma_priors : array-like, shape (n_samples,)

    X : array-like, shape (n_samples, n_features)

    location : array-like, shape (n_features,)

    reg_scale : float

    Returns
    -------
    scales : array, shape (n_features,)
        The scale matrix of the distribution.
    """
    n_samples = X.shape[0]
    avg_X2 = np.dot(gamma_priors, X * X)
    avg_location2 = gamma_priors.sum() * (location * location)
    avg_X_location = location * np.dot(gamma_priors, X)
    return (avg_X2 - 2 * avg_X_location + avg_location2) / n_samples + reg_scale


def _estimate_student_scales_spherical(gamma_priors, X, location, reg_scale):
    """Estimate the spherical scale value.

    Parameters
    ----------
    gamma_priors : array-like, shape (n_samples,)

    X : array-like, shape (n_samples, n_features)

    location : array-like, shape (n_features,)

    reg_scale : float

    Returns
    -------
    scales : scalar
        The scale matrix of the distribution.
    """
    return _estimate_student_scales_diag(gamma_priors, X,
                                         location, reg_scale).mean()


def _estimate_student_parameters(X, gamma_priors, reg_scale, scale_type):
    """Estimate the Student's t-distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    gamma_priors : array-like, shape (n_samples,)
        The gamma weights for each data sample in X.

    reg_scale : float
        The regularization added to the diagonal of the scale matrix.

    scale_type : {'full', 'diag', 'spherical'}
        The type of scale matrix.

    Returns
    -------
    location : array-like, shape (n_features,)
        The centers of the distribution.

    scale : array-like
        The scale matrix of the distribution.
        The shape depends of the scale_type.
    """
    location = np.dot((gamma_priors).T, X) / (gamma_priors).T.sum()
    scale = {"full": _estimate_student_scales_full,
             "diag": _estimate_student_scales_diag,
             "spherical": _estimate_student_scales_spherical
             }[scale_type](gamma_priors, X, location, reg_scale)
    return location, scale


def _compute_precision_cholesky(scale, scale_type):
    """Compute the Cholesky decomposition of the precision matrix.

    Parameters
    ----------
    scale : array-like
        The scale matrix of the distribution.
        The shape depends of the scale_type.

    scale_type : {'full', 'diag', 'spherical'}
        The type of scale matrix.

    Returns
    -------
    precision_cholesky : array-like
        The cholesky decomposition of the precision matrix of the
        distribution. The shape depends of the scale_type.
    """
    estimate_precision_error_message = (
        "Fitting the t-distribution failed because it has an ill-defined "
        "empirical scale (for instance caused by singleton or collapsed"
        "samples). Try to increase reg_scale.")

    if scale_type == 'full':
        try:
            cov_chol = linalg.cholesky(scale, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        n_features = scale.shape[0]
        precision_chol = linalg.solve_triangular(cov_chol,
                                                     np.eye(n_features),
                                                     lower=True).T
    else:
        if np.any(np.less_equal(scale, 0.0)):
            print(scale)
            raise ValueError(estimate_precision_error_message)
        precision_chol = 1. / np.sqrt(scale)
    return precision_chol


def _compute_gamma_priors(X, location, precision_chol, scale_type, dof):
    """Estimate the gamma weights of samples.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    location : array-like, shape (n_features,)

    precision_chol : array-like
        Cholesky decompositions of the precision matrix.
        'full' : shape of (n_features, n_features)
        'diag' : shape of (n_features,)
        'spherical' : scalar

    scale_type : {'full', 'diag', 'spherical'}

    dofs : scalar

    Returns
    -------
    gamma_weights : array, shape (n_samples,)
    """

    _, n_features = X.shape

    if scale_type == 'full':
        y = np.dot(X, precision_chol) - np.dot(location, precision_chol)
        prob = np.sum(np.square(y), axis=1)

    elif scale_type == 'diag':
        precision = precision_chol ** 2
        prob = (np.sum((location ** 2 * precision)) -
                2. * np.dot(X, (location * precision).T) +
                np.dot(X ** 2, precision))

    else:
        precision = precision_chol ** 2
        prob = (np.sum(location ** 2) * precision -
                2 * np.dot(X, location.T * precision) +
                precision * row_norms(X, squared=True))

    return (dof + n_features) / (dof + prob)


def _compute_delta(X, location, precision_chol, scale_type):
    """Compute the delta (distance from center) of samples.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    location : array-like, shape (n_features,)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrix.
        'full' : shape of (n_features, n_features)
        'diag' : shape of (n_features,)
        'spherical' : scalar

    scale_type : {'full', 'diag', 'spherical'}

    Returns
    -------
    delta : array, shape (n_samples,)
    """   
    if scale_type == 'full':
        y = np.dot(X, precision_chol) - np.dot(location, precision_chol)
        return np.sum(np.square(y), axis=1)

    elif scale_type == 'diag':
        precision = precision_chol ** 2
        return (np.sum((location ** 2 * precision)) -
                2. * np.dot(X, (location * precision).T) +
                np.dot(X ** 2, precision))

    else:
        precision = precision_chol ** 2
        return (np.sum(location ** 2) * precision -
                2 * np.dot(X, location.T * precision) +
                precision * row_norms(X, squared=True))


###############################################################################
# Student mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, scale_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrix.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_features, n_features)
        'diag' : shape of (n_features,)
        'spherical' : scalar

    scale_type : {'full', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix.
    """
    if scale_type == 'full':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif scale_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol)))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_student_prob(X, location, precision_chol, scale_type, dof):
    """Estimate the log Student probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    location : array-like, shape (n_features,)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrix.
        'full' : shape of (n_features, n_features)
        'diag' : shape of (n_features,)
        'spherical' : scalar

    scale_type : {'full', 'diag', 'spherical'}

    dof : scalar

    Returns
    -------
    log_prob : array, shape (n_samples,)
    """
    _, n_features = X.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(precision_chol, scale_type, n_features)

    if scale_type == 'full':
        y = np.dot(X, precision_chol) - np.dot(location, precision_chol)
        prob = np.sum(np.square(y), axis=1)

    elif scale_type == 'diag':
        precision = precision_chol ** 2
        prob = (np.sum((location ** 2 * precision)) -
                2. * np.dot(X, (location * precision).T) +
                np.dot(X ** 2, precision.T))

    else:
        precision = precision_chol ** 2
        prob = (np.sum(location ** 2) * precision -
                2 * np.dot(X, location * precision) +
                precision * row_norms(X, squared=True))
    log_prob = np.log(1 + (prob / dof))
    return gammaln(.5 * (dof + n_features)) - gammaln(.5 * dof) - .5 * (n_features * np.log(dof * np.pi) +
                                                                        (dof + n_features) * log_prob) + log_det


def _initialize_dof(X, location, precision_chol, scale_type, max_iter=100, tol=1e-3):
    """Initial degrees-of-freedom estimation

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    location : array-like, shape (n_features,)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrix.
        'full' : shape of (n_features, n_features)
        'diag' : shape of (n_features,)
        'spherical' : scalar

    scale_type : {'full', 'diag', 'spherical'}

    max_iter : int, defaults to 100.
        The number of iterations to perform for degrees-of-freedom estimation.

    tol : float, defaults to 1e-3.
        The degrees-of-freedom estimation convergence threshold.

    Returns
    -------
    dof : scalar
    """

    n_samples, n_features = X.shape
    delta = _compute_delta(X, location, precision_chol, scale_type)
    dof = 30
    def init_function(df):
        sample_dependent_component = -(np.log(df + delta) + 1.0 * (df + n_features) / (df + delta))
        gamma_component = digamma(1.0 * (df + n_features) / 2.0) - digamma(1.0 * df / 2.0)
        log_component = 1.0 * np.log(df) + 1.0
        return gamma_component + log_component + (1.0 / n_samples) * (
            np.sum(sample_dependent_component)
        )

    def init_first_derivative(df):
        sample_dependent_component = -(1.0 / (df + delta) + 1.0 * (n_features - delta) / ((df + delta) ** 2))
        gamma_component = (1.0 / 2.0) * (polygamma(1, 1.0 * (df + n_features) / 2.0) - polygamma(1, 1.0 * df / 2.0))
        log_component = 1.0 / df
        return gamma_component + log_component - (1 / n_samples) * (
            np.sum(sample_dependent_component)
        )

    def init_second_derivative(df):
        sample_dependent_component = 1.0 / ((df + delta) ** 2) + 2.0 * (n_features - delta) / ((df + delta) ** 3)
        gamma_component = (1.0 / 4.0) * (polygamma(2.0, 1.0 * (df + n_features) / 2.0) - polygamma(2, 1.0 * df / 2.0))
        log_component = -1.0 / (df ** 2)
        return gamma_component + log_component + (1.0 / n_samples) * (
            np.sum(sample_dependent_component)
        )
    for i in range(3, 31):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                new_dof = newton(
                    init_function,
                    i,
                    init_first_derivative,
                    args=(),
                    maxiter=max_iter,
                    tol=tol,
                    fprime2=init_second_derivative,
                    full_output=False,
                    disp=False,
                )
                if np.isfinite(new_dof) and new_dof > 2:
                    return new_dof

        except Exception:
            pass
    return dof


def _estimate_dof(gamma_priors, current_dof, n_features, max_iter=100, tol=1e-3):
    """Estimate the degrees of freedom

    Parameters
    ----------
    gamma_priors : array-like, shape (n_samples,)
        The gamma priors for each data sample in X.

    current_dof : scalar
        Current degrees-of-freedom estimation.

   n_features : int
        Number of features.

    max_iter : int, defaults to 100.
        The number of iterations to perform for degrees-of-freedom estimation.

    tol : float, defaults to 1e-3.
        The degrees-of-freedom estimation convergence threshold.

    Returns
    -------
    dof : scalar
    """
    n_samples = gamma_priors.shape[0]
    constant = (
        1.0
        + (1.0 / n_samples)
        * (np.log(gamma_priors) - gamma_priors).sum()
        + digamma(1.0 * (current_dof + n_features) / 2.0)
        - np.log(1.0 * (current_dof + n_features) / 2.0)
    )

    def function(df):
        return -digamma(1.0 * df / 2.0) + np.log(1.0 * df / 2.0) + constant

    def first_derivative(df):
        return -(1.0 / 2.0) * polygamma(1, 1.0 * df / 2.0) + 1.0 / df

    def second_derivative(df):
        return -(1.0 / 4.0) * polygamma(2, 1.0 * df / 2.0) - 1.0 / (df * df)

    dof = newton(
        function,
        current_dof,
        first_derivative,
        args=(),
        maxiter=max_iter,
        tol=tol,
        fprime2=second_derivative,
        full_output=False,
        disp=False,
    )

    return dof
