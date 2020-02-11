import numpy as np

from scipy import linalg
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import newton

from sklearn.utils.extmath import row_norms
from sklearn.utils import check_array


###############################################################################

# Student mixture shape checkers used by the StudentMixture class
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


def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
    """Check the input data X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    n_components : int

    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_locations(locations, n_components, n_features):
    """Validate the provided 'locations'.

    Parameters
    ----------
    locations : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    locations : array, (n_components, n_features)
    """
    locations = check_array(locations, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(locations, (n_components, n_features), 'locations')
    return locations


def _check_dofs(dofs, n_components):
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
    dofs = check_array(dofs, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(dofs, (n_components,), 'dofs')

    # check range
    if any(np.less(dofs, 2)):
        raise ValueError("The parameter 'dofs' should be in the range "
                         "[2, inf), but got min value %.5f"
                         % (np.min(dofs)))
    return dofs


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


def _check_precisions_full(precisions, scale_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, scale_type)


def _check_precisions(precisions, scale_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    scale_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False,
                             allow_nd=scale_type == 'full')

    precisions_shape = {'full': (n_components, n_features, n_features),
                        'tied': (n_features, n_features),
                        'diag': (n_components, n_features),
                        'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[scale_type],
                 '%s precision' % scale_type)

    _check_precisions_type = {'full': _check_precisions_full,
                              'tied': _check_precision_matrix,
                              'diag': _check_precision_positivity,
                              'spherical': _check_precision_positivity}
    _check_precisions_type[scale_type](precisions, scale_type)
    return precisions


###############################################################################
# Student mixture parameters estimators (used by the M-Step)

def _estimate_student_scales_full(resp, gamma_priors, X, nk, locations, reg_scale):
    """Estimate the full scale matrices.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)
    
    gamma_priors : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    locations : array-like, shape (n_components, n_features)

    reg_scale : float

    Returns
    -------
    scales : array, shape (n_components, n_features, n_features)
        The scale matrix of the current components.
    """
    n_components, n_features = locations.shape
    scales = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - locations[k]
        scales[k] = np.dot(resp[:, k] * gamma_priors[:, k] * diff.T, diff) / nk[k]
        scales[k].flat[::n_features + 1] += reg_scale
    return scales


def _estimate_student_scales_tied(resp, gamma_priors, X, nk, locations, reg_scale):
    """Estimate the tied scale matrix.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)
    
    gamma_priors : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    locations : array-like, shape (n_components, n_features)

    reg_scale : float

    Returns
    -------
    scale : array, shape (n_features, n_features)
        The tied scale matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_locations2 = np.dot(nk * locations.T, locations)
    scale = avg_X2 - avg_locations2
    scale /= nk.sum()
    scale.flat[::len(scale) + 1] += reg_scale
    return scale


def _estimate_student_scales_diag(resp, gamma_priors, X, nk, locations, reg_scale):
    """Estimate the diagonal scale vectors.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)
    
    gamma_priors : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    locations : array-like, shape (n_components, n_features)

    reg_scale : float

    Returns
    -------
    scales : array, shape (n_components, n_features)
        The scale vector of the current components.
    """
    avg_X2 = np.dot((resp * gamma_priors).T, X * X) / nk[:, np.newaxis]
    avg_locations2 = locations ** 2
    avg_X_locations = locations * np.dot((resp * gamma_priors).T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_locations + avg_locations2 + reg_scale


def _estimate_student_scales_spherical(resp, gamma_priors, X, nk, locations, reg_scale):
    """Estimate the spherical variance values.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)
    
    gamma_priors : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    locations : array-like, shape (n_components, n_features)

    reg_scale : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_student_scales_diag(resp, gamma_priors, X, nk,
                                         locations, reg_scale).mean(1)


def _estimate_student_parameters(X, resp, gamma_priors, reg_scale, scale_type):
    """Estimate the Student's t-distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    gamma_priors : array-like, shape (n_samples, n_components)
        The gamma priors for each data sample in X.

    reg_scale : float
        The regularization added to the diagonal of the scale matrices.

    scale_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    locations : array-like, shape (n_components, n_features)
        The centers of the current components.

    scales : array-like
        The scale matrix of the current components.
        The shape depends on scale_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    locations = np.dot((resp * gamma_priors).T, X) / (resp * gamma_priors).T.sum(1)[:, np.newaxis]
    scales = {"full": _estimate_student_scales_full,
              "tied": _estimate_student_scales_tied,
              "diag": _estimate_student_scales_diag,
              "spherical": _estimate_student_scales_spherical
              }[scale_type](resp, gamma_priors, X, nk, locations, reg_scale)
    return nk, locations, scales


def _compute_precision_cholesky(scales, scale_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    scales : array-like
        The scale matrix of the current components.
        The shape depends of the scale_type.

    scale_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the scale_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical scale (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_scale.")

    if scale_type in 'full':
        n_components, n_features, _ = scales.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, scale in enumerate(scales):
            try:
                cov_chol = linalg.cholesky(scale, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif scale_type == 'tied':
        _, n_features = scales.shape
        try:
            cov_chol = linalg.cholesky(scales, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(scales, 0.0)):
            print(scales)
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(scales)
    return precisions_chol


def _compute_gamma_priors(X, locations, precisions_chol, scale_type, dofs):
    """Estimate the gamma weights of samples.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    locations : array-like, shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    scale_type : {'full', 'tied', 'diag', 'spherical'}

    dofs : array-like, shape (n_components,)

    Returns
    -------
    gamma_weights : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = locations.shape

    if scale_type == 'full':
        prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(locations, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            prob[:, k] = np.sum(np.square(y), axis=1)

    elif scale_type == 'tied':
        prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(locations):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            prob[:, k] = np.sum(np.square(y), axis=1)

    elif scale_type == 'diag':
        precisions = precisions_chol ** 2
        prob = (np.sum((locations ** 2 * precisions), 1) -
                2. * np.dot(X, (locations * precisions).T) +
                np.dot(X ** 2, precisions.T))

    else:
        precisions = precisions_chol ** 2
        prob = (np.sum(locations ** 2, 1) * precisions -
                2 * np.dot(X, locations.T * precisions) +
                np.outer(row_norms(X, squared=True), precisions))

    return (dofs + n_features) / (dofs + prob)


###############################################################################
# Student mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, scale_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    scale_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if scale_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif scale_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif scale_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_student_prob(X, locations, precisions_chol, scale_type, dofs):
    """Estimate the log Student probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    locations : array-like, shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    scale_type : {'full', 'tied', 'diag', 'spherical'}

    dofs : array-like, shape (n_components,)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = locations.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, scale_type, n_features)

    if scale_type == 'full':
        prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(locations, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            prob[:, k] = np.sum(np.square(y), axis=1)

    elif scale_type == 'tied':
        prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(locations):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            prob[:, k] = np.sum(np.square(y), axis=1)

    elif scale_type == 'diag':
        precisions = precisions_chol ** 2
        prob = (np.sum((locations ** 2 * precisions), 1) -
                2. * np.dot(X, (locations * precisions).T) +
                np.dot(X ** 2, precisions.T))

    else:
        precisions = precisions_chol ** 2
        prob = (np.sum(locations ** 2, 1) * precisions -
                2 * np.dot(X, locations.T * precisions) +
                np.outer(row_norms(X, squared=True), precisions))
    log_prob = np.log(1 + (prob / dofs))
    return (gammaln(0.5*(dofs + n_features)) - gammaln(0.5 * dofs) + log_det -
            0.5 * (n_features * np.log(dofs * np.pi) + (dofs + n_features) * log_prob))


def _initialize_dofs(X, scales, scale_type, n_components, resp, gamma_priors, n_features, max_iter=100, tol=1e-3):
    """Initializes the degrees-of-freedom.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    scales : array-like
        scale matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    scale_type : {'full', 'tied', 'diag', 'spherical'}

    n_components : int
        Number of components.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    gamma_priors : array-like, shape (n_samples, n_components)
        The gamma priors for each data sample in X.

    n_features : int
        Number of features.

    max_iter : int, defaults to 100.
        The number of iterations to perform for degrees-of-freedom estimation.

    tol : float, defaults to 1e-3.
        The degrees-of-freedom estimation convergence threshold.

    Returns
    -------
    dofs : array-like, shape (n_components,)
        Initial degrees-of-freedom estimation.
    """
    arg_resp = resp.argmax(1)
    current_dofs = np.zeros((n_components,))
    for k in range(n_components):
        covariance = np.cov(X[arg_resp == k, :].T)
        if scale_type == 'full':
            scale = np.diag(scales[k, :, :])
        elif scale_type == 'tied':
            scale = np.diag(scales)
        elif scale_type == 'diag':
            scale = scales[k, :]
        else:
            scale = np.array(n_features * [scales[k]])
        dof_estimation = 2 / (1 - np.mean(scale * np.diag(np.linalg.pinv(covariance))))
        current_dofs[k] = np.minimum(np.maximum(2, dof_estimation), 30)
    return _estimate_dofs(resp, gamma_priors, current_dofs, n_features, max_iter=max_iter, tol=tol)


def _estimate_dofs(resp, gamma_priors, current_dofs, n_features, max_iter=100, tol=1e-3):
    """Estimates the degrees-of-freedom.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    gamma_priors : array-like, shape (n_samples, n_components)
        The gamma priors for each data sample in X.

    current_dofs : array-like, shape (n_components,)
        Current degrees-of-freedom estimation.

    n_features : int
        Number of features.

    max_iter : int, defaults to 100.
        The number of iterations to perform for degrees-of-freedom estimation.

    tol : float, defaults to 1e-3.
        The degrees-of-freedom estimation convergence threshold.

    Returns
    -------
    dofs : array-like, shape (n_components,)
        The degrees-of-freedom estimation.
    """
    n_components = current_dofs.shape[0]
    dofs = np.zeros((n_components,))
    for k in range(n_components):
        constant = (
                1.0
                + (1.0 / resp[:, k].sum(axis=0))
                * (resp[:, k] * (np.log(gamma_priors[:, k]) - gamma_priors[:, k])).sum(axis=0)
                + digamma(1.0 * (current_dofs[k] + n_features) / 2.0)
                - np.log(1.0 * (current_dofs[k] + n_features) / 2.0)
        )

        def function(df):
            return -digamma(1.0 * df / 2.0) + np.log(1.0 * df / 2.0) + constant

        def first_derivative(df):
            return -(1.0 / 2.0) * polygamma(1, 1.0 * df / 2.0) + 1.0 / df

        def second_derivative(df):
            return -(1.0 / 4.0) * polygamma(2, 1.0 * df / 2.0) - 1.0 / (df * df)

        dofs[k] = newton(
            function,
            current_dofs[k],
            first_derivative,
            args=(),
            maxiter=max_iter,
            tol=tol,
            fprime2=second_derivative,
            full_output=False,
            disp=False,
        )

    return dofs
