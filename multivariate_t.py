#
# Author: Omri Tomer 2020
#
from __future__ import division, print_function, absolute_import

import numpy as np

from scipy.special import gammaln, digamma

from scipy.stats._multivariate import (multi_rv_generic, multi_rv_frozen,
                                       multivariate_normal, _PSD, _squeeze_output)
from _multivariate_t_functions import _multivariate_t_random, _multivariate_t_cdf

_LOG_2PI = np.log(2 * np.pi)
_LOG_PI = np.log(np.pi)


class multivariate_t_gen(multi_rv_generic):
    """
    A multivariate Student's t-distribution random variable.
    The `location` keyword specifies the location. The `scale` keyword specifies the
    scale matrix. The 'dof' keyword specifies the degrees-of-freedom.

    Methods
    -------
    ``pdf(x, location=None, scale=1, dof=None, allow_singular=False)``
        Probability density function.
    ``logpdf(x, location=None, scale=1, dof=None, allow_singular=False)``
        Log of the probability density function.
    ``cdf(x, location=None, scale=1, dof=None, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Cumulative distribution function.
    ``logcdf(x, location=None, scale=1, dof=None, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Log of the cumulative distribution function.
    ``rvs(location=None, scale=1, dof=None, size=1, random_state=None)``
        Draw random samples from a multivariate t-distribution.
    ``entropy()``
        Compute the differential entropy of the multivariate Student's.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate Student's
    random variable:
    rv = multivariate_t(location=None, scale=1, dof=None, allow_singular=False)
        - Frozen object with the same methods but holding the given
          mean and covariance fixed.

    Notes
    -----
    The scale matrix `scale` must be a (symmetric) positive
    semi-definite matrix. The determinant and inverse of `scale` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `scale` does not need to have full rank.
    The probability density function for `multivariate_t` is
    .. math::
        f(x) = \frac{\Gamma[(\nu+p)/2]}{\Gamma(\nu/2)\sqrt{(\pi \nu)^p |\Sigma|}}
               \left(1+\frac{1}{\nu} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)^{-(\nu+p)/2}
    where :math:`\mu` is the location, :math:`\Sigma` the scale matrix,
    :math:`\nu` is the degrees-of-freedom, and :math:`k` is the
    dimension of the space where :math:`x` takes values.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from student_mixture import multivariate_t
    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_t.pdf(x, location=2.5, scale=0.5, dof=30); y
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> fig1 = plt.figure()
    >>> ax = fig1.add_subplot(111)
    >>> ax.plot(x, y)
    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:
    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_t([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]], 20)
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))
    """

    def __init__(self, seed=None):
        super(multivariate_t_gen, self).__init__(seed)

    def __call__(self, location=None, scale=1, dof=None, allow_singular=False, seed=None):
        """
        Create a frozen multivariate t distribution.
        See `multivariate_t_frozen` for more information.
        """
        return multivariate_t_frozen(location, scale, dof,
                                     allow_singular=allow_singular,
                                     seed=seed)

    def _process_parameters(self, dim, location, scale, dof):
        """
        Infer dimensionality from location or scale matrix, ensure that
        location and scale are full vector resp. matrix, and ensure that
        2 < dof.
        """

        # Try to infer dimensionality
        if dim is None:
            if location is None:
                if scale is None:
                    dim = 1
                else:
                    scale = np.asarray(scale, dtype=float)
                    if scale.ndim < 2:
                        dim = 1
                    else:
                        dim = scale.shape[0]
            else:
                location = np.asarray(location, dtype=float)
                dim = location.size
        else:
            if not np.isscalar(dim):
                raise ValueError("Dimension of random variable must be "
                                 "a scalar.")

        # Check input sizes and return full arrays for location
        # and scale if necessary
        if location is None:
            location = np.zeros(dim)
        location = np.asarray(location, dtype=float)

        if scale is None:
            scale = 1.0
        scale = np.asarray(scale, dtype=float)

        if dim == 1:
            location.shape = (1,)
            scale.shape = (1, 1)

        if location.ndim != 1 or location.shape[0] != dim:
            raise ValueError("Array 'location' must be a vector of length %d." %
                             dim)
        if scale.ndim == 0:
            scale = scale * np.eye(dim)
        elif scale.ndim == 1:
            scale = np.diag(scale)
        elif scale.ndim == 2 and scale.shape != (dim, dim):
            rows, cols = scale.shape
            if rows != cols:
                msg = ("Array 'scale' must be square if it is two dimensional,"
                       " but scale.shape = %s." % str(scale.shape))
            else:
                msg = ("Dimension mismatch: array 'scale' is of shape %s,"
                       " but 'location' is a vector of length %d.")
                msg = msg % (str(scale.shape), len(location))
            raise ValueError(msg)
        elif scale.ndim > 2:
            raise ValueError("Array 'scale' must be at most two-dimensional,"
                             " but scale.ndim = %d" % scale.ndim)
        if dof is None:
            dof = np.inf
        elif dof > 2:
            pass
        else:
            raise ValueError("Parameter 'dof' must be >2,"
                             " but dof = %d" % dof)

        return dim, location, scale, dof

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

    def _logpdf(self, x, location, prec_U, log_det_scale, rank, dof):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        location : ndarray
            Location of the distribution
        prec_U : ndarray
            A decomposition such that np.dot(prec_U, prec_U.T)
            is the precision matrix, i.e. inverse of the scale matrix.
        log_det_cov : float
            Logarithm of the determinant of the scale matrix
        rank : int
            Rank of the scale matrix.
        dof : scalar
            Degrees-of-freedom of the distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.
        """
        if dof == np.inf:
            return multivariate_normal._logpdf(x, location, prec_U, log_det_scale, rank)
        else:
            gamma_ratio = gammaln((dof + rank) / 2) - gammaln(dof / 2)
            if gamma_ratio != np.nan:
                dev = x - location
                log_maha = np.log(1 + np.sum(np.square(np.dot(dev, prec_U)), axis=-1) / dof)
                return gamma_ratio - 0.5 * (rank * (_LOG_PI + np.log(dof)) + log_det_scale + (dof + rank) * log_maha)
            else:
                return multivariate_normal._logpdf(x, location, prec_U, log_det_scale, rank)

    def logpdf(self, x, location=None, scale=1, dof=None, allow_singular=False):
        """
        Log of the multivariate Student's t probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        location : ndarray
            Location of the distribution
        scale : array_like
            Scale matrix of the distribution
        dof : scalar
            Degrees-of-freedom of the distribution

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`
        """
        dim, location, scale, dof = self._process_parameters(None, location, scale, dof)
        x = self._process_quantiles(x, dim)
        psd = _PSD(scale, allow_singular=allow_singular)
        out = self._logpdf(x, location, psd.U, psd.log_pdet, psd.rank, dof)
        return _squeeze_output(out)

    def pdf(self, x, location=None, scale=1, dof=None, allow_singular=False):
        """
        Multivariate Student's t probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
                location : ndarray
            Location of the distribution
        scale : array_like
            Scale matrix of the distribution
        dof : scalar
            Degrees-of-freedom of the distribution

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`
        """
        dim, location, scale, dof = self._process_parameters(None, location, scale, dof)
        x = self._process_quantiles(x, dim)
        psd = _PSD(scale, allow_singular=allow_singular)
        out = np.exp(self._logpdf(x, location, psd.U, psd.log_pdet, psd.rank, dof))
        return _squeeze_output(out)

    def _cdf(self, x, location, scale, dof, maxpts, abseps, releps):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        location : ndarray
            Location of the distribution
        scale : array_like
            Scale matrix of the distribution
        dof : scalar
            Degrees-of-freedom of the distribution
        maxpts: integer
            The maximum number of points to use for integration (only used when dim > 3)
        abseps: float
            Absolute error tolerance (only used when dim > 1)
        releps: float
            Relative error tolerance (only used when dim == 2 or dim == 3)

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.
        """
        if dof == np.inf:
            return multivariate_normal.cdf(x, location, scale, maxpts=maxpts, abseps=abseps, releps=releps)
        else:
            return _multivariate_t_cdf(x, location, scale, dof, maxpts=maxpts, abseps=abseps, releps=releps)

    def logcdf(self, x, location=None, scale=1, dof=None, allow_singular=False, maxpts=None,
               abseps=1e-5, releps=1e-5):
        """
        Log of the multivariate Student's t cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        location : ndarray
            Location of the distribution
        scale : array_like
            Scale matrix of the distribution
        dof : scalar
            Degrees-of-freedom of the distribution
        maxpts: integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance (default 1e-5)
        releps: float, optional
            Relative error tolerance (default 1e-5)

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`
        """
        dim, location, scale, dof = self._process_parameters(None, location, scale, dof)
        x = self._process_quantiles(x, dim)
        # Use _PSD to check covariance matrix
        _PSD(scale, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * dim
        out = np.log(self._cdf(x, location, scale, dof, maxpts, abseps, releps))
        return out

    def cdf(self, x, location=None, scale=1, dof=None, allow_singular=False, maxpts=None,
            abseps=1e-5, releps=1e-5):
        """
        Multivariate Student's t cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        location : ndarray
            Location of the distribution
        scale : array_like
            Scale matrix of the distribution
        dof : scalar
            Degrees-of-freedom of the distribution
        maxpts: integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance (default 1e-5)
        releps: float, optional
            Relative error tolerance (default 1e-5)

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`
        """
        dim, location, scale, dof = self._process_parameters(None, location, scale, dof)
        x = self._process_quantiles(x, dim)
        # Use _PSD to check covariance matrix
        _PSD(scale, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * dim
        out = self._cdf(x, location, scale, dof, maxpts, abseps, releps)
        return out

    def rvs(self, location=None, scale=1, dof=None, size=1, random_state=None):
        """
        Draw random samples from a multivariate Student's t distribution.

        Parameters
        location : ndarray
            Location of the distribution
        scale : array_like
            Scale matrix of the distribution
        dof : scalar
            Degrees-of-freedom of the distribution
        size : int
            Number of samples to draw
        random_state : np.random.RandomState, optional

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.
        """
        _, location, scale, dof = self._process_parameters(None, location, scale, dof)
        if dof == np.inf:
            random_state = self._get_random_state(random_state)
            out = random_state.multivariate_normal(location, scale, size)
            return _squeeze_output(out)
        else:
            return _multivariate_t_random(location, scale, dof, size, random_state)

    def entropy(self, location=None, scale=1, dof=None):
        """
        Compute the differential entropy of the multivariate Student's t.

        Parameters
        ----------
        location : ndarray
            Location of the distribution
        scale : array_like
            Scale matrix of the distribution
        dof : scalar
            Degrees-of-freedom of the distribution

        Returns
        -------
        h : scalar
            Entropy of the multivariate Student's t distribution

        Notes
        -----
        References:

        """
        if dof == np.inf:
            return multivariate_normal.entropy(location, scale)
        else:
            dim, location, scale, dof = self._process_parameters(None, location, scale, dof)
            _, logdet = np.linalg.slogdet(scale)
            loggamma_ratio = gammaln(0.5 * dof) - gammaln(0.5 * (dof + dim)) + (0.5 * dim) * (np.log(dof) + _LOG_PI)
            digamma_part = (0.5 * (dof + dim)) * (digamma(0.5 * (dof + dim)) - digamma(0.5 * dof))
            return 0.5 * logdet + loggamma_ratio + digamma_part


multivariate_t = multivariate_t_gen()


class multivariate_t_frozen(multi_rv_frozen):
    def __init__(self, location=None, scale=1, dof=None, allow_singular=False, seed=None,
                 maxpts=None, abseps=1e-5, releps=1e-5):
        """
        Create a frozen multivariate Student's t-distribution.

        Parameters
        ----------
        location : array_like, optional
            Location of the distribution (default zero)
        scale : array_like, optional
            Scale matrix of the distribution (default one)
        dof : None or scalar, optional
            Degrees-of-freedom of the distribution (default numpy.inf)
        allow_singular : bool, optional
            If this flag is True then tolerate a singular
            scale matrix (default False).
        seed : None or int or np.random.RandomState instance, optional
            This parameter defines the RandomState object to use for drawing
            random variates.
            If None (or np.random), the global np.random state is used.
            If integer, it is used to seed the local RandomState instance
            Default is None.
        maxpts: integer, optional
            The maximum number of points to use for integration of the
            cumulative distribution function (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance for the cumulative distribution function
            (default 1e-5)
        releps: float, optional
            Relative error tolerance for the cumulative distribution function
            (default 1e-5)

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:
        >>> from student_mixture import multivariate_t
        >>> r = multivariate_t()
        >>> r.location
        array([ 0.])
        >>> r.scale
        array([[1.]])
        >>> r.dof
        inf
        """

        self._dist = multivariate_t_gen(seed)
        self.dim, self.location, self.scale, self.dof = self._dist._process_parameters(
                                                            None, location, scale, dof)
        self.scale_info = _PSD(self.scale, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.location, self.scale_info.U,
                                 self.scale_info.log_pdet, self.scale_info.rank, self.dof)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logcdf(self, x):
        return np.log(self.cdf(x))

    def cdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(x, self.location, self.scale, self.dof, self.maxpts, self.abseps,
                              self.releps)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.location, self.scale, self.dof, size, random_state)

    def entropy(self):
        """
        Computes the differential entropy of the multivariate Student's t.

        Returns
        -------
        h : scalar
            Entropy of the multivariate Student's t distribution
        """
        if self.dof == np.inf:
            return multivariate_normal.entropy(self.location, self.scale)
        else:
            return self._dist.entropy(self.location, self.scale, self.dof)
