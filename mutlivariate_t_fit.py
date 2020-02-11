"""Student's t-distribution Fitting."""

# Author: Omri Tomer <omritomer1@mail.tau.ac.il>
# License: BSD 3 clause

import numpy as np

from scipy import linalg

from sklearn.utils.validation import check_is_fitted

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.fixes import logsumexp

from _multivariate_t_fit_functions import (_check_X, _check_location, _check_dof, _check_precision,
                                            _estimate_student_parameters, _compute_precision_cholesky,
                                            _compute_gamma_priors, _estimate_log_student_prob,
                                            _initialize_dof, _estimate_dof)
from multivariate_t import multivariate_t


class MultivariateTFit():
    """Multivariate Studnet's t-distribution fitting object.

    Class to fit a multivariate Student's t-distribution.
    This class allows estimation of the parameters of a
    multivariate Student's t-distribution.

    Parameters
    ----------
    scale_type : {'full' (default), 'diag', 'spherical'}
        String describing the type of scale parameters to use.
        Must be one of:

        'full'
            full scale matrix
        'diag'
            diagonal scale matrix
        'spherical'
            single variance

    algorithm : {'em' (default), 'mcecm'}
        String describing the algorithm used for estimating
        the degrees-of-freedom.
        Must be one of:

        'em'
            Expectation-Maximization algorithm
        'mcecm'
            Multicycle Expectation-Conditional-Maximization algorithm

    fixed_dof : boolean, default to False.
        Determines whether the degrees-of-freedom are estimated
        or fixed. If fixed, then the default value for 'dof_init'
        is np.inf

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_scale : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of scale.
        Allows to assure that the scale matrix is positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    init_params : {'gaussian', 'random'}, defaults to 'gaussian'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'gaussian' : responsibilities are initialized using Gaussian distribution.
            'random' : responsibilities are initialized randomly.

    dof_tol : float, defaults to 1e-3.
        The degrees-of-freedom estimation convergence threshold.

    dof_max_iter : int, defaults to 100.
        The number of iterations to perform for degrees-of-freedom estimation.

    location : array-like, shape (n_features,), optional
        The user-provided initial location, defaults to None,
        If it None, location is initialized using the `init_params` method.

    precision_init : array-like, optional.
        The user-provided initial precisions (inverse of the scale
        matrix), defaults to None.
        If it None, precision is initialized using the 'init_params' method.
        The shape depends on 'scale_type'::

            (1,)                        if 'spherical',
            (n_features,)               if 'diag',
            (n_features, n_features)    if 'full'

    dof_init : scalar (2 <= dof < inf), optional
        The user-provided initial degrees-of freedom, defaults to None.
        If None, dof is initialized using the `init_params` method.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    output : boolean, default to False.
        Determines the method 'fit' returns a multivariate_t
        object with the estimated parameters

    Attributes
    ----------
    location_ : array-like, shape (n_features,)
        The location (mean) of the t-distribution.

    scale_ : array-like
        The scale of the t-distribution.
        The shape depends on `scale_type`::

            (1,)                        if 'spherical',
            (n_features,)               if 'diag',
            (n_features, n_features)    if 'full'

    dof_ : scalar (2 <= dof_ < inf)
        The degrees-of-freedom of the t-distribution.

    precision_ : array-like
        The precision matrix of the t-distribution. A precision
        matrix is the inverse of the scale matrix. The scale matrix is
        symmetric positive definite so the Student's t-distribution
        can be equivalently parameterized by the precision matrix.
        Storing the precision matrix instead of the scale matrix makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `scale_type`::

            (1,)                        if 'spherical',
            (n_features,)               if 'diag',
            (n_features, n_features)    if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrix of the
        distribution. A precision matrix is the inverse of a scale matrix.
        A scale matrix is symmetric positive definite the Student's
        t-distributions can be equivalently parameterized by the
        precision matriX.
        Storing the precision matrix instead of the scale matrix makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `scale_type`::

            (1,)                        if 'spherical',
            (n_features,)               if 'diag',
            (n_features, n_features)    if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.


    """

    def __init__(self, scale_type='full', algorithm='em',
                 fixed_dof=False,
                 tol=1e-3, reg_scale=1e-6, max_iter=100,
                 dof_tol=1e-3, dof_max_iter=100,
                 location_init=None, precision_init=None, dof_init=None,
                 random_state=None,
                 verbose=0, verbose_interval=10,
                 output=False):

        self.scale_type = scale_type
        self.algorithm = algorithm
        self.fixed_dof = fixed_dof
        self.dof_tol = dof_tol
        self.dof_max_iter = dof_max_iter
        self.location_init = location_init
        self.precision_init = precision_init
        self.dof_init = dof_init
        self.tol = tol
        self.reg_scale = reg_scale
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.output = output

    def _check_parameters(self, X):
        """Check that the parameters are well defined."""
        n_features = X.shape[1]
        if self.scale_type not in ['spherical', 'diag', 'full']:
            raise ValueError("Invalid value for 'scale_type': %s "
                             "'scale_type' should be in "
                             "['spherical', 'diag', 'full']"
                             % self.scale_type)

        if self.location_init is not None:
            self.location_init = _check_location(self.location_init, n_features)

        if self.precision_init is not None:
            self.precision_init = _check_precision(self.precision_init,
                                                   self.scale_type,
                                                   n_features)
        if self.dof_init is not None:
            self.dof_init = _check_dof(self.dof_init)

    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """

        if self.algorithm not in ['em', 'mcecm']:
            raise ValueError("Invalid value for 'algorithm': %s "
                             "'algorithm' should be in "
                             "['em', 'mcecm']"
                             % self.algorithm)

        if not isinstance(self.fixed_dof, bool):
            raise TypeError("Invalid value for 'fixed_dof': %s "
                            "'fixed_dof' must be boolean: "
                            "True or False"
                            % self.fixed_dof)
        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        if self.reg_scale < 0.:
            raise ValueError("Invalid value for 'reg_scale': %.5f "
                             "regularization on scale must be "
                             "non-negative"
                             % self.reg_scale)

        if not isinstance(self.output, bool):
            raise TypeError("Invalid value for 'output': %s "
                            "'output' must be boolean: "
                            "True or False"
                            % self.output)

        # Check all the parameters values of the derived class
        self._check_parameters(X)

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        The method fits the model. The method iterates between E-step and
        M-step for ``max_iter`` times until the change of likelihood or
        lower bound is less than ``tol``, otherwise, a ``ConvergenceWarning``
        is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self.fit_predict(X, y)

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model. The method iterates between E-step and
        M-step for ``max_iter`` times until the change of likelihood or
        lower bound is less than ``tol``, otherwise, a ``ConvergenceWarning``
        is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        """
        X = _check_X(X, ensure_min_samples=2)
        self._check_initial_parameters(X)

        self.converged_ = False

        self._initialize(X)

        lower_bound = -np.infty

        for n_iter in range(1, self.max_iter + 1):
            prev_lower_bound = lower_bound

            log_prob_norm, gamma_priors = self._e_step(X)
            self._m_step(X, gamma_priors)
            lower_bound = self._compute_lower_bound(log_prob_norm)

            change = lower_bound - prev_lower_bound

            if abs(change) < self.tol:
                self.converged_ = True
                break

        if not self.converged_:
            warnings.warn('Initialization did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.',
                          ConvergenceWarning)

        self._set_parameters(self._get_parameters())
        self.n_iter_ = n_iter
        self.lower_bound_ = lower_bound
        if self.output:
            return self._dist

    def _initialize(self, X):
        """Initialization of the Student's t-distribution parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        n_samples, _ = X.shape
        location, scale = _estimate_student_parameters(
            X, np.ones(n_samples), self.reg_scale, self.scale_type)

        self.location_ = location if self.location_init is None else self.location_init
        if self.precision_init is None:
            self.scale_ = scale
            self.precision_cholesky_ = _compute_precision_cholesky(
                scale, self.scale_type)
        elif self.scale_type == 'full':
            self.precision_cholesky_ = linalg.cholesky(self.precision_init, lower=True)
        elif self.scale_type == 'tied':
            self.precision_cholesky_ = linalg.cholesky(self.precision_init, lower=True)
        else:
            self.precision_cholesky_ = self.precision_init

        self.dof_ = _initialize_dof(X, self.location_, self.precision_cholesky_,self.scale_type
                                    ) if self.dof_init is None else self.dof_init

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        gamma_priors : array, shape (n_samples,)
            Gamma weights of each sample in X.
        """
        log_prob_norm, gamma_priors = self._estimate_log_prob_gamma(X)
        self.gamma_priors_ = gamma_priors
        return log_prob_norm, gamma_priors

    def _estimate_log_prob_gamma(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probability, and the prior Gamma weightsof the samples
        in X with respect to the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        gamma_priors : array, shape (n_samples,)
            Gamma weights of each sample in X.
        """
        log_prob = self._estimate_log_prob(X)
        log_prob_norm = logsumexp(log_prob)
        gamma_priors = self._estimate_gamma_priors(X)
        return log_prob_norm, gamma_priors

    def _estimate_gamma_priors(self, X):
        """Estimate the gamma priors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        gamma_priors : array, shape (n_samples,)
            Gamma weights of each sample in X.
        """
        return _compute_gamma_priors(X, self.location_, self.precision_cholesky_, self.scale_type, self.dof_)

    def _m_step(self, X, gamma_priors):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        gamma_priors : array, shape (n_samples,)
            Gamma weights of each sample in X.
        """
        _, n_features = X.shape
        self.location_, self.scale_ = (
            _estimate_student_parameters(X, gamma_priors, self.reg_scale,
                                          self.scale_type))

        if not self.fixed_dof:
            if self.algorithm == 'mcecm':
                _, gamma_priors = self._e_step(X)
            self.dof_ = _estimate_dof(gamma_priors, self.dof_, n_features)

        self.precision_cholesky_ = _compute_precision_cholesky(
            self.scale_, self.scale_type)

    def _estimate_log_prob(self, X):
        """Estimate the log-probability of the model."""
        return _estimate_log_student_prob(
            X, self.location_, self.precision_cholesky_, self.scale_type, self.dof_)

    def _compute_lower_bound(self, log_prob_norm):
        """Returns the lower bound for the EM algorithm."""
        return log_prob_norm

    def _check_is_fitted(self):
        """Check that the model is fitted and the parameters have been set."""
        check_is_fitted(self, ['dof_', 'location_', 'precision_cholesky_'])

    def _get_parameters(self):
        """Get the parameters of the model."""
        return (self.location_, self.scale_,
                self.precision_cholesky_, self.dof_)

    def _set_parameters(self, params):
        """Set the parameters of the model."""
        (self.location_, self.scale_,
         self.precision_cholesky_, self.dof_) = params
        self._dist = multivariate_t(self.location_, self.scale_, self.dof_)

        if self.scale_type == 'full':
            self.precision_ = np.dot(self.precision_cholesky_, self.precision_cholesky_.T)
        else:
            self.precision_ = self.precision_cholesky_ ** 2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        n_features = self.location_.shape[0]
        if self.scale_type == 'full':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.scale_type == 'diag':
            cov_params = n_features
        else:
            cov_params = 1
        return int(cov_params + n_features + 1)

    def pdf(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return self._dist.pdf(X)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.log_likelihood(X) +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.log_likelihood(X) + 2 * self._n_parameters()

    def log_likelihood(self, X):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_likelihood : float
            Log likelihood of the distribution given X.
        """
        return self._dist.logpdf(X).sum()

    def rvs(self, n_samples=1):
        """Generate random samples from the fitted Multivariate t-distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        """
        self._check_is_fitted()

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (n_samples))

        rng = check_random_state(self.random_state)

        return self._dist.rvs(size=n_samples, random_state=rng)
