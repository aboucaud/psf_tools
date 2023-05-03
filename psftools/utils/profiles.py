"""
PSF profiles submodule
----------------------
Common PSF profiles and helper functions

"""
import numpy as np

from psftools.utils.misc import fwhm2sigma

__all__ = ["moffat", "gaussian", "psf_moffat", "psf_gaussian", "circ_gauss2d"]


# Main methods
# ------------
def moffat(radius, fwhm, beta):
    """
    Moffat optical PSF profile

    Parameters
    ----------
    radius: `numpy.ndarray`
        Input array of radial coordinates
    fwhm: float
        Full width at half-maximum
    beta: float
        Slope of the profile

    Returns
    -------
    profile: `numpy.ndarray`
        Radial profile corresponding to the input coordinates

    """
    theta0 = 0.5 * fwhm
    alpha = (2 ** (1 / beta) - 1) / theta0**2
    profile = (beta - 1) / np.pi * alpha * np.power(1 + alpha * radius**2, -1 * beta)

    return profile


def gaussian(radius, sigma):
    """
    Basic gaussian function with zero mean

    Parameters
    ----------
    radius: `numpy.ndarray`
        Input array of radial coordinates
    sigma: float
        Standard deviation of the Gaussian profile

    Returns
    -------
    profile: `numpy.ndarray`
        Radial profile corresponding to the input coordinates

    """
    profile = 1 / (2 * np.pi * sigma**2) * np.exp(-0.5 * np.power(radius / sigma, 2))

    return profile


def psf_moffat(radius, fwhm):
    """
    PSF derived in Racine (1996)

    Parameters
    ----------
    radius: `numpy.ndarray`
        Input array of radial coordinates
    fwhm: float
        Full width at half-maximums

    Returns
    -------
    profile: `numpy.ndarray`
        Radial profile corresponding to the input coordinates

    """
    return 0.8 * moffat(radius, fwhm, 7) + 0.2 * moffat(radius, fwhm, 2)


def psf_gaussian(radius, fwhm):
    """
    Gaussian optical PSF

    Parameters
    ----------
    radius: `numpy.ndarray`
        Input array of radial coordinates
    fwhm: float
        Full width at half-maximum

    Returns
    -------
    profile: `numpy.ndarray`
        Radial profile corresponding to the input coordinates

    """
    sigma = fwhm2sigma(fwhm)
    return gaussian(radius, sigma)


def circ_gauss2d(shape, xc, yc, sigma):
    """
    Circular 2d Gaussian function

    Parameters
    ----------
    shape: tuple of floats
        Shape of the output array
    xc: float
        Center along the x-axis (column)
    yc: float
        Center along the y-axis (row)
    sigma: float
        Width of the Gaussian profile

    Returns
    -------
    output: `numpy.ndarray`
        Image of a circurlar 2d Gaussian function

    """
    y, x = np.indices(shape)
    exponent = 0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / sigma**2
    return np.exp(-exponent) / (2 * sigma**2)
