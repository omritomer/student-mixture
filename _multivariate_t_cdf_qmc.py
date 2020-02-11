import numpy as np
from scipy.special import erfc, erfcinv, gammaincinv
from _halton import halton_sample


def _multivariate_t_cdf_single_sample(x, chol, dof, tol=1e-4, max_evaluations=1e+7):
    """
    Wrapper function for estimating the cdf of single sample 'x'.

    Parameters
    ----------
    x : array_like
        Standardized sample, shape (n_features,)
    chol : array_like, shape (n_features, n_features)
        Cholesky decomposition of the correlation matrix of the distribution
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    tol: float
        Tolerance for quasi-Monte Carlo algorithm
    max_evaluations: float
        Maximum points to evaluate with quasi-Monte Carlo algorithm


    Returns
    -------
    cdf : scalar
        Cumulative density function evaluated at `x`
    """
    if np.all(x == np.inf):
        return 1, 0

    elif np.all(x == -np.inf):
        return 0, 0

    else:
        return _multivariate_t_cdf_quasi_monte_carlo(x, chol, dof, tol=tol, max_evaluations=max_evaluations)


def _multivariate_t_cdf_quasi_monte_carlo(x, chol, dof, tol=1e-4, max_evaluations=1e+7, n_repetitions=30):
    """
    Function for estimating the cdf of single sample 'x'.
    Algorithm based on:
    Genz, A. and F. Bretz (1999) "Numerical Computation of Multivariate
    t Probabilities with Application to Power Calculation of Multiple
    Contrasts", J.Statist.Comput.Simul., 63:361-378.
    Genz, A. and F. Bretz (2002) "Comparison of Methods for the
    Computation of Multivariate t Probabilities", J.Comp.Graph.Stat.,
    11(4):950-971.

    Parameters
    ----------
    x : array_like
        Standardized sample, shape (n_features,)
    chol : array_like, shape (n_features, n_features)
        Cholesky decomposition of the correlation matrix of the distribution
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    tol: float
        Tolerance for quasi-Monte Carlo algorithm
    max_evaluations: float
        Maximum points to evaluate with quasi-Monte Carlo algorithm


    Returns
    -------
    cdf : scalar
        Cumulative density function evaluated at `x`
    """
    # Prime list generated using student_mixture._generate_prime.gen_selected_prime_list
    primes = [31, 47, 71, 107, 163, 241, 359, 541, 811, 1217, 1823, 2731, 4099, 6151, 9227,
              13841, 20759, 31139, 46703, 70051, 105071, 157627, 236449, 354677, 532009, 798023,
              1197037, 1795559, 2693329, 4039991, 6059981, 9089981]

    dim = chol.shape[0]
    c = np.diag(chol)
    x = x / c
    chol = chol / np.tile(c, (dim, 1))

    prime = primes[0]
    fun_evaluations = 2 * n_repetitions * prime
    t, sigma_squared = mvt_qmc(x, chol, dof, n_repetitions, prime, dim)
    err = 3.5 * np.sqrt(sigma_squared)

    for i in range(1, len(primes)):
        prime = primes[i]
        fun_evaluations += 2 * n_repetitions * prime
        if fun_evaluations > max_evaluations:
            break

        t_hat, sigma_squared_tau_hat = mvt_qmc(x, chol, dof, n_repetitions, prime, dim)

        t += sigma_squared * (t_hat - t) / (sigma_squared + sigma_squared_tau_hat)
        sigma_squared *= sigma_squared_tau_hat / (sigma_squared_tau_hat + sigma_squared)
        err = 3.5 * np.sqrt(sigma_squared)

        if err < tol:
            return t, err
    return t, err


def mvt_qmc(x, chol, dof, mc_repetitions, prime, dim):
    p = halton_sample(prime, dim)  # draw quasi-random lattice points from halton sequence
    t_hat = np.zeros(mc_repetitions)

    for rep in range(mc_repetitions):
        w = np.tile(np.random.uniform(low=0.0, high=1.0, size=dim), (prime, 1))
        v = abs(2 * ((p + w) % 1) - 1)
        t_hat[rep] = f_quasi_variable_separation(x, chol, dof, v, prime, dim)

    return np.mean(t_hat), np.var(t_hat) / mc_repetitions


def f_quasi_variable_separation(x, chol, dof, v, n_points, dim):
    return 0.5 * (f_variable_separation(x, chol, dof, v, n_points, dim) +
                  f_variable_separation(x, chol, dof, 1 - v, n_points, dim))


def f_variable_separation(x, chol, dof, v, n_points, dim, tail_tol=np.finfo(float).eps):
    s = chiinv(v[:, -1], dof) / (dof ** 0.5)
    e = normcdf(s * x[0])
    t = 1.0 * e
    y = np.zeros((n_points, dim))

    for i in range(1, dim):
        z = np.maximum(np.minimum(e * v[:, -1 - i], 1 - tail_tol), tail_tol)
        y[:, i - 1] = normicdf(z)
        e = normcdf(s * x[i] - np.dot(y, chol[:, i]))
        t *= e

    return t.mean()


def normcdf(x):
    return 0.5 * erfc(-x / (2 ** 0.5))


def normicdf(x):
    return -(2 ** 0.5) * erfcinv(2 * x)


def chiinv(x, nu):
    return (2 * gammaincinv(nu / 2, x)) ** 0.5
