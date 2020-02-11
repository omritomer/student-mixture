import numpy as np
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.special import betainc
from _multivariate_t_cdf_qmc import _multivariate_t_cdf_single_sample


###############################################################################
# Functions that estimate the cumulative distribution function of the standard multivariate t-distribution


def _standard_univariate_t_cdf(x, dof):
    """
    Standard univariate Student's t cumulative density function.
    Reference:
    Johnson, N. L., Kotz, S., Balakrishnan, N. ( 1995 ). Continuous
    Univariate Distributions. Vol. 2, 2nd ed. New York : Wiley.
    Parameters
    ----------
    x : array_like
        Standardized samples, shape (n_samples, )
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    Returns
    -------
    cdf : ndarray or scalar
        Cumulative density function evaluated at `x`
    """
    sign = np.sign(x)
    sign[sign == 0] = 1
    f_positive = 1.0 - 0.5 * betainc(dof / 2.0, 0.5, dof / (dof + x ** 2))

    return (sign < 0) + sign * f_positive


def _standard_bivariate_t_cdf(x, corr_mat, dof, abseps=1e-8, releps=1e-8):
    """
    Standard bivariate Student's t cumulative density function.
    Algorithm based on:
    Genz, A. (2004). Numerical computation of rectangular bivariate
    and trivariate normal and t probabilities. Statistics and
    Computing, 14(3), 251-260.

    Parameters
    ----------
    x : array_like
        Standardized samples, shape (n_samples, 2)
    corr_mat : array_like, shape (2, 2)
        Correlation matrix of the distribution,must be symmetric and positive
        definite, with all elements of the diagonal being 1
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    abseps: float
        Absolute error tolerance for integration
    releps: float
        Relative error tolerance for integration
    Returns
    -------
    cdf : ndarray or scalar
        Cumulative density function evaluated at `x`
    """

    n = x.shape[0]
    rho = corr_mat[0, 1]
    if rho >= 0:
        tau_s = _standard_univariate_t_cdf(x.min(axis=1), dof)
    else:
        tau_s = np.maximum(_standard_univariate_t_cdf(x[:, 0], dof) - _standard_univariate_t_cdf(-x[:, 1], dof), 0)

    lower_bound = np.sign(rho) * np.pi / 2.0 if rho != 0 else np.pi / 2.0
    upper_bound = np.arcsin(rho)

    integral = np.array(
        [quad(bivariate_integrand, lower_bound, upper_bound,
              (x[i, 0], x[i, 1], dof),
              epsabs=abseps, epsrel=releps)[0] if np.all(np.isfinite(x[i, :]))
         else 0.0 for i in range(n)])

    return tau_s + integral / (2 * np.pi)


def _standard_trivariate_t_cdf(x, corr_mat, dof, abseps=1e-8, releps=1e-8):
    """
    Standard trivariate Student's t cumulative density function.
    Algorithm based on:
    Genz, A. (2004). Numerical computation of rectangular bivariate
    and trivariate normal and t probabilities. Statistics and
    Computing, 14(3), 251-260.

    Parameters
    ----------
    x : array_like
        Standardized samples, shape (n_samples, 3)
    corr_mat : array_like, shape (3, 3)
        Correlation matrix of the distribution,must be symmetric and positive
        definite, with all elements of the diagonal being 1
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    abseps: float
        Absolute error tolerance for integration
    releps: float
        Relative error tolerance for integration
    Returns
    -------
    cdf : ndarray or scalar
        Cumulative density function evaluated at `x`
    """

    n = x.shape[0]
    rho = corr_mat.ravel()[[1, 2, 5]]
    (rho_21, rho_31, rho_32) = rho

    if rho_32 >= 0:
        tau_r_double_star = _standard_bivariate_t_cdf(np.array([x[:, 0], x[:, 1:].min(axis=1)]).T,
                                                      np.eye(2), dof, abseps=abseps, releps=releps)
    else:
        tau_r_double_star_1 = _standard_bivariate_t_cdf(x[:, 0:2],
                                                        np.eye(2), dof, abseps=abseps, releps=releps)
        tau_r_double_star_2 = _standard_bivariate_t_cdf(np.array([x[:, 0], -x[:, 2]]).T,
                                                        np.eye(2), dof, abseps=abseps, releps=releps)
        tau_r_double_star = np.maximum(tau_r_double_star_1 - tau_r_double_star_2, 0)

    lower_bound_integral_1 = np.sign(rho_32) * np.pi / 2.0 if rho_32 != 0 else np.pi / 2.0
    upper_bound_integral_1 = np.arcsin(rho_32)
    lower_bound_integral_2 = 0
    upper_bound_integral_2 = np.arcsin(rho_21)
    lower_bound_integral_3 = 0
    upper_bound_integral_3 = np.arcsin(rho_31)

    integral_1 = np.array(
        [quad(trivariate_integrand1, lower_bound_integral_1, upper_bound_integral_1,
              (x[i, 0], x[i, 1], x[i, 2], dof),
              epsabs=abseps, epsrel=releps)[0] if np.all(np.isfinite(x[i, :]))
         else 0.0 for i in range(n)])
    integral_2 = np.array(
        [quad(trivariate_integrand2, lower_bound_integral_2, upper_bound_integral_2,
              (x[i, 0], x[i, 1], x[i, 2], rho_21, rho_31, rho_32, dof),
              epsabs=abseps, epsrel=releps)[0] if np.all(np.isfinite(x[i, :]))
         else 0.0 for i in range(n)]) if abs(rho_21) > 0 else 0.0
    integral_3 = np.array(
        [quad(trivariate_integrand2, lower_bound_integral_3, upper_bound_integral_3,
              (x[i, 0], x[i, 2], x[i, 1], rho_31, rho_21, rho_32, dof),
              epsabs=abseps, epsrel=releps)[0] if np.all(np.isfinite(x[i, :]))
         else 0.0 for i in range(n)]) if abs(rho_31) > 0 else 0.0

    return tau_r_double_star + (integral_1 + integral_2 + integral_3) / (2 * np.pi)


def _standard_multivariate_t_cdf(x, corr_mat, dof, tol=1e-4, max_evaluations=1e+7):
    """
    Wrapper function for _multivariate_t_cdf_single_sample
    See multivariate_t_cdf_qmc for more on algorithm

    Parameters
    ----------
    x : array_like
        Standardized samples, shape (n_samples, n_features)
    corr_mat : array_like, shape (n_features, n_features)
        Correlation matrix of the distribution,must be symmetric and positive
        definite, with all elements of the diagonal being 1
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    tol: float
        Tolerance for quasi-Monte Carlo algorithm
    max_evaluations: float
        Maximum points to evaluate with quasi-Monte Carlo algorithm

    Returns
    -------
    cdf : ndarray or scalar
        Cumulative density function evaluated at `x`
    """
    n = x.shape[0]
    cdf = np.zeros(n)
    err = np.zeros(n)
    chol = cholesky(corr_mat)
    for i in range(n):
        cdf[i], err[i] = _multivariate_t_cdf_single_sample(x[i, :], chol, dof, tol=tol, max_evaluations=max_evaluations)
    return cdf


###############################################################################
# Integral functions


def bivariate_integrand(theta, b1, b2, nu):
    sin_theta = np.sin(theta)
    cos_theta_squared = np.cos(theta) ** 2
    return (1 + ((b1 ** 2) + (b2 ** 2) - 2 * sin_theta * b1 * b2)
            / (nu * cos_theta_squared)) ** (-nu / 2.0)


def trivariate_integrand1(theta, b1, b2, b3, nu):
    sin_theta = np.sin(theta)
    cos_theta_squared = 1 - sin_theta ** 2
    y = (1 + tvt_f_i(sin_theta, cos_theta_squared, b2, b3) / nu) ** (-0.5)
    return (y ** nu) * tcdf(b1 * y, nu)


def trivariate_integrand2(theta, b1, b_j, b_k, rho_j1, rho_k1, rho_kj, nu):
    sin_theta = np.sin(theta)
    cos_theta_squared = 1 - sin_theta ** 2
    y = (1 + tvt_f_i(sin_theta, cos_theta_squared, b1, b_j) / nu) ** (-0.5)
    u_hat = tvt_u_hat_k(sin_theta, cos_theta_squared, b1, b_j, b_k, rho_j1, rho_k1, rho_kj)
    return (y ** nu) * tcdf(u_hat * y, nu)


def tvt_f_i(sin_theta, cos_theta_squared, b_j, b_k):
    return (b_j ** 2 + b_k ** 2 - 2 * sin_theta * b_j * b_k) / cos_theta_squared


def tvt_u_hat_k(sin_theta, cos_theta_squared, b1, b_j, b_k, rho_j1, rho_k1, rho_kj):
    sin_theta_div_rho_j1 = sin_theta / rho_j1
    sin_theta_squared_div_rho_j1 = sin_theta * sin_theta_div_rho_j1
    sin_theta_squared_times_rho_k1_div_rho_j1 = rho_k1 * sin_theta_squared_div_rho_j1
    u_numerator = (b_k * cos_theta_squared -
                   b1 * sin_theta_div_rho_j1 * (rho_k1 - rho_j1 * rho_kj) -
                   b_j * (rho_kj - sin_theta_squared_times_rho_k1_div_rho_j1))
    u_denominator = (cos_theta_squared *
                     (cos_theta_squared - (rho_k1 * sin_theta_div_rho_j1) ** 2 -
                      rho_kj ** 2 + 2 * sin_theta_squared_times_rho_k1_div_rho_j1 * rho_kj))
    return u_numerator / (u_denominator ** 0.5)


def tcdf(x, dof):
    f = 0.5 * betainc(dof / 2.0, 0.5, dof / (dof + x ** 2))
    return 1 - f if x >= 0 else f
