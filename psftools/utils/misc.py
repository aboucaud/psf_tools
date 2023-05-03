"""
Misc submodule
---------------
Helper methods for use in various contexts.

"""
import numpy as np

__all__ = ["print_help", "cart2pol", "pol2cart", "sigma2fwhm", "fwhm2sigma"]


def print_help(method, message, *args, **kwargs):
    """Print wrapper to help debugging"""
    main_msg = "PSF tools | {}: {}"
    return print(main_msg.format(method, message), *args, **kwargs)


def cart2pol(x, y, unit="deg"):
    """Transform cartesian coordinates to polar

    Parameters
    ----------
    x : float or `numpy.ndarray`
        x-axis coordinate(s)
    y : float or `numpy.ndarray`
        y-axis coordinate(s)
    unit : str, optional
        Angle unit (default degrees 'deg')

    Returns
    -------
    theta : float or `numpy.ndarray`
        Polar angle
    rho : float or `numpy.ndarray`
        Polar distance

    """
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    if unit == "deg":
        return rho, np.rad2deg(theta)

    return theta, rho


def pol2cart(theta, rho, unit="deg"):
    """Transform polar coordinates to cartesian

    Parameters
    ----------
    theta : float or `numpy.ndarray`
        Polar angle
    rho : float or `numpy.ndarray`
        Polar distance
    unit : str, optional
        Angle unit (default degrees 'deg')

    Returns
    -------
    x : float or `numpy.ndarray`
        x-axis coordinate(s)
    y : float or `numpy.ndarray`
        y-axis coordinate(s)

    """
    if unit == "deg":
        theta = np.deg2rad(theta)

    return rho * np.cos(theta), rho * np.sin(theta)


def sigma2fwhm(sigma):
    """Convert the gaussian width to a fwhm"""
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    return fwhm


def fwhm2sigma(fwhm):
    """Convert the fwhm to a gaussian width"""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))
