"""Student's t-distribution Mixture Model."""

# Author: Omri Tomer <omritomer1@mail.tau.ac.il>
# License: BSD 3 clause

import numpy as np

from scipy import linalg
from scipy.special import logsumexp

from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted

import warnings

from sklearn import cluster
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.fixes import logsumexp

from _student_mixture_functions import (_check_weights, _check_X, _check_locations, _check_dofs, _check_precisions,
                                         _estimate_student_parameters, _compute_precision_cholesky,
                                         _compute_gamma_priors, _estimate_log_student_prob,
                                         _initialize_dofs, _estimate_dofs)
from _multivariate_t_functions import _multivariate_t_random, _multivariate_t_cdf


class StudentMixture(GaussianMixture):
    """Student Mixture.

    Representation of a Student's t- mixture model probability distribution.
    This class allows to estimate the parameters of a Student's mixture
    distribution.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    scale_type : {'full' (default), 'tied', 'diag', 'spherical'}
        String describing the type of scale parameters to use.
        Must be one of:

        'full'
            each component has its own general scale matrix
        'tied'
            all components share the same general scale matrix
        'diag'
            each component has its own diagonal scale matrix
        'spherical'
            each component has its own single variance

    algorithm : {'em' (default), 'mcecm'}
        String describing the algorithm used for estimating
        the degrees-of-freedom.
        Must be one of:

        'em'
            Expectation-Maximization algorithm
        'mcecm'
            Multicycle Expectation-Conditional-Maximization algorithm (not recommended)

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_scale : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of scale.
        Allows to assure that the scale matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the locations and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using k-means.
            'random' : responsibilities are initialized randomly.

    dof_tol : float, defaults to 1e-3.
        The degrees-of-freedom estimation convergence threshold.

    dof_max_iter : int, defaults to 100.
        The number of iterations to perform for degrees-of-freedom estimation.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    locations_init : array-like, shape (n_components, n_features), optional
        The user-provided initial locations, defaults to None,
        If it None, locations are initialized using the `init_params` method.

    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the scale
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'scale_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    dofs_init : array-like, shape (n_components, ), optional
        The user-provided initial degrees-of-freedom, defaults to None (2 <= dofs_init[i] < inf for all i).
        If it None, dofs are initialized using the `init_params` method.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    locations_ : array-like, shape (n_components, n_features)
        The location of each mixture component.

    scales_ : array-like
        The scale of each mixture component.
        The shape depends on `scale_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    dofs_ : array-like, shape (n_components,)
        The degrees-of-freedom of each mixture components.

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a scale matrix. A scale matrix is
        symmetric positive definite so the mixture of Student's t-distributions
        can be equivalently parameterized by the precision matrices. Storing
        the precision matrices instead of the scale matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `scale_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a scale matrix.
        A scale matrix is symmetric positive definite so the mixture of
        Student's t-distributions can be equivalently parameterized by the
        precision matrices.
        Storing the precision matrices instead of the scale matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `scale_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.


    """

    def __init__(self, n_components=1, scale_type='full',
                 algorithm='em', fixed_dofs=False,
                 tol=1e-3, reg_scale=1e-6,
                 max_iter=100, n_init=1, init_params='kmeans',
                 dof_tol=1e-3, dof_max_iter=100,
                 weights_init=None, locations_init=None, precisions_init=None, dofs_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_scale,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.scale_type = scale_type
        self.algorithm = algorithm
        self.reg_scale = reg_scale
        self.fixed_dofs = fixed_dofs
        self.dof_tol = dof_tol
        self.dof_max_iter = dof_max_iter
        self.weights_init = weights_init
        self.locations_init = locations_init
        self.precisions_init = precisions_init
        self.dofs_init = dofs_init

    def _check_parameters(self, X):
        """Check the Student mixture parameters are well defined."""
        _, n_features = X.shape
        if self.scale_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'scale_type': %s "
                             "'scale_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.scale_type)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.locations_init is not None:
            self.locations_init = _check_locations(self.locations_init,
                                                   self.n_components,
                                                   n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.scale_type,
                                                     self.n_components,
                                                     n_features)

        if self.dofs_init is not None:
            self.dofs_init = _check_dofs(self.dofs_init,
                                         self.n_components)

    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        if self.reg_scale < 0.:
            raise ValueError("Invalid value for 'reg_scale': %.5f "
                             "regularization on scale must be "
                             "non-negative"
                             % self.reg_scale)
                             
        super()._check_initial_parameters(X)

        if self.algorithm not in ['em', 'mcecm']:
            raise ValueError("Invalid value for 'algorithm': %s "
                             "'algorithm' should be in "
                             "['em', 'mcecm']"
                             % self.algorithm)

        if not isinstance(self.fixed_dofs, bool):
            raise TypeError("Invalid value for 'fixed_dofs': %s "
                            "'fixed_dofs' must be boolean: "
                            "True or False"
                            % self.fixed_dofs)

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)
        gamma_priors = np.ones(resp.shape)

        self._initialize(X, resp, gamma_priors)

    def _initialize(self, X, resp, gamma_priors):
        """Initialization of the Student mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)

        gamma_priors : array-like, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape

        weights, locations, scales = _estimate_student_parameters(
            X, resp, gamma_priors, self.reg_scale, self.scale_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.locations_ = (locations if self.locations_init is None
                           else self.locations_init)
        self.dofs_ = _initialize_dofs(
            X, scales, self.scale_type, self.n_components,
            resp, gamma_priors, n_features
        ) if self.dofs_init is None else self.dofs_init

        if self.precisions_init is None:
            self.scales_ = scales
            self.precisions_cholesky_ = _compute_precision_cholesky(
                scales, self.scale_type)
        elif self.scale_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.scale_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised. After fitting, it
        predicts the most probable label for the input data points.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = _check_X(X, self.n_components, ensure_min_samples=2)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = (-np.infty if do_init else self.lower_bound_)

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp, gamma_priors = self._e_step(X)
                self._m_step(X, log_resp, gamma_priors)
                lower_bound = self._compute_lower_bound(
                    log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp, _ = self._e_step(X)

        return log_resp.argmax(axis=1)
        
    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            location of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        gamma_priors : array, shape (n_samples, n_components)
            Gamma weights of the point of each sample in X.
        """
        log_prob_norm, log_resp, gamma_priors = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp, gamma_priors

    def _m_step(self, X, log_resp, gamma_priors):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        gamma_priors : array-like, shape (n_samples, n_components)
            Gamma weights of the point of each sample in X.
        """
        n_samples, n_features = X.shape
        resp = np.exp(log_resp)
        self.weights_, self.locations_, self.scales_ = (
            _estimate_student_parameters(X, resp, gamma_priors, self.reg_scale,
                                         self.scale_type))
        if not self.fixed_dofs:
            if self.algorithm == 'mcemc':
                _, log_resp, gamma_priors = self._e_step(X)
            self.dofs_ = _estimate_dofs(np.exp(log_resp), gamma_priors,
                                        self.dofs_, n_features)
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.scales_, self.scale_type)

    def _estimate_log_prob(self, X):
        """Estimate the log-probability of the model."""
        return _estimate_log_student_prob(
            X, self.locations_, self.precisions_cholesky_,
            self.scale_type, self.dofs_)

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities

        gamma_priors : array, shape (n_samples, n_components)
            Gamma weights of the point of each sample in X.
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        gamma_priors = self._estimate_gamma_priors(X)
        return log_prob_norm, log_resp, gamma_priors

    def _estimate_gamma_priors(self, X):
        """Estimate the gamma priors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        gamma_priors : array, shape (n_samples, n_component)
        """
        return _compute_gamma_priors(X, self.locations_, self.precisions_cholesky_, self.scale_type, self.dofs_)

    def _estimate_log_weights(self):
        """Estimate the logarithm of the weights of the model."""
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        """Returns the lower bound for the EM algorithm."""
        return log_prob_norm

    def _check_is_fitted(self):
        """Check that the model is fitted and the parameters have been set."""
        check_is_fitted(self, ['weights_', 'locations_', 'precisions_cholesky_', 'dofs_'])

    def _get_parameters(self):
        """Get the parameters of the model."""
        return (self.weights_, self.locations_, self.scales_,
                self.precisions_cholesky_, self.dofs_)

    def _set_parameters(self, params):
        """Set the parameters of the model."""
        (self.weights_, self.locations_, self.scales_,
         self.precisions_cholesky_, self.dofs_) = params

        # Attributes computation

        if self.scale_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.scale_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.locations_.shape
        if self.scale_type == 'full':
            scale_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.scale_type == 'diag':
            scale_params = self.n_components * n_features
        elif self.scale_type == 'tied':
            scale_params = n_features * (n_features + 1) / 2.
        else:
            scale_params = self.n_components
        location_params = n_features * self.n_components
        return int(scale_params + location_params + 2 * self.n_components - 1)

    def pdf(self, X):
        """Probability distribution function of the samples in X

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        pdf : array of shape (n_samples,)
        """
        return np.exp(self.score_samples(X))

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

    def predict_proba(self, X):
        """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability of each mixture (state) in
            the model given each sample.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.locations_.shape[1])
        _, log_resp, _ = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.locations_.shape[1])

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.locations_.shape[1])
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Student's t mixture distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        y : array, shape (nsamples,)
            Component labels

        """
        self._check_is_fitted()

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % self.n_components)

        _, n_features = self.locations_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        if self.scale_type == 'full':
            X = np.vstack([
                _multivariate_t_random(location, scale, dof, int(sample), rng)
                for (location, scale, dof, sample) in zip(
                    self.locations_, self.scales_, self.dofs_, n_samples_comp)])
        elif self.scale_type == "tied":
            X = np.vstack([
                _multivariate_t_random(location, self.scales_, dof, int(sample), rng)
                for (location, dof, sample) in zip(
                    self.locations_, n_samples_comp)])
        elif self.scale_type == 'diag':
            X = np.vstack([
                _multivariate_t_random(location, np.diag(scale), dof, int(sample), rng)
                for (location, scale, dof, sample) in zip(
                    self.locations_, self.scales_, self.dofs_, n_samples_comp)])
        else:
            X = np.vstack([
                _multivariate_t_random(location, np.diag(self.scales_.repeat(n_features)), dof, int(sample), rng)
                for (location, dof, sample) in zip(
                    self.locations_, n_samples_comp)])

        y = np.concatenate([np.full(sample, j, dtype=int)
                           for j, sample in enumerate(n_samples_comp)])

        return X, y

    def rvs(self, n_samples=1):
        """Generate random samples from the fitted Student's t mixture distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        """
        X, _ = self.sample(n_samples)

        return X

    def cdf(self, X, maxpts=1e+7, abseps=1e-6, releps=1e-6):
        """Cumulative distribution function of the samples in X

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        maxpts: integer
            The maximum number of points to use for integration (used when n_features > 3)
        abseps: float
            Absolute error tolerance (used when n_features > 1)
        releps: float
            Relative error tolerance (used when n_features == 2 or n_features == 3)

        Returns
        -------
        cdf : array of shape (n_samples,)
        """

        self._check_is_fitted()
        X = _check_X(X, None, self.locations_.shape[1])
        n_features = X.shape[1]
        f = np.zeros(X.shape[0])

        for i in range(self.n_components):
            location = self.locations_[i, :]
            dof = self.dofs_[i]
            weight = self.weights_[i]
            if self.scale_type == 'full':
                scale = self.scales_[i, :, :]
            elif self.scale_type == "tied":
                scale = self.scales_
            elif self.scale_type == 'diag':
                scale = np.diag(self.scales_[i, :])
            else:
                scale = np.diag(np.array([self.scales_[i]] * n_features))

            f = f + weight * _multivariate_t_cdf(X, location, scale, dof, maxpts=maxpts, abseps=abseps, releps=releps)

        f[f > 1.0] = 1.0
        f[f < 0.0] = 0.0
        return f
