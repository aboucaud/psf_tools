"""
Analysis submodule
------------------

"""
import numpy as np

__all__ = ["get_radius", "profile", "get_radius_old", "integrated_profile", "circ_psd"]


def slow_get_radius(value, distance, profile):
    """Return the radius corresponding to the input profile value

    Parameters
    ----------
    value: float
        Profile value which the radius is wanted
    distance: `numpy.ndarray`
        Radius grid
    profile: `numpy.ndarray`
        Image profile

    Returns
    -------
    radius: float
        Distance at which the profile matches the input value
        (same dimensionality of the distance input array)

    Remarks
    -------
    Only works for a monotonious profile

    """
    if value >= profile.max():
        return distance.min()

    if value <= profile.min():
        return distance.max()

    scaled_profile = profile - value
    idx = np.argmin(np.abs(scaled_profile))
    if scaled_profile[idx] < 0:
        idmin = idx - 1
        idmax = idx
    elif scaled_profile[idx] > 0:
        idmin = idx
        idmax = idx + 1
    else:
        return distance[idx]
    delta = (value - profile[idmin]) / (profile[idmax] - profile[idmin])

    return distance[idmin] + delta * (distance[idmax] - distance[idmin])


def slow_azimuthal_profile(image, origin, bin, nbins):
    """Azimuthal profile of a 2d image.

    x, y[, n] = profile(image, [origin, bin, nbins, histogram])

    Parameters
    ----------
    input: `numpy.ndarray`
        2D input array
    origin: tuple of float
        Center of profile (Fits convention)
    bin: float
        Width of the profile bins (in unit of pixels).
    nbins: int
        Number of profile bins.

    Returns
    -------
    radius: `numpy.ndarray`
        Profile radius in unit of pixels
    profile: `numpy.ndarray`
        Profile of input array
    histo: `numpy.ndarray`
        Histogram of pixel count in each bin

    """
    nx = image.shape[0]
    ny = image.shape[1]
    xmid, ymid = origin

    rad = np.zeros(nbins, dtype=np.float)
    profile = np.zeros(nbins, dtype=np.float)
    histo = np.zeros(nbins, dtype=np.int)
    for i in range(nx):
        for j in range(ny):
            val = image[i, j]
            if np.isnan(val):
                continue
            distance = np.sqrt((i - xmid) ** 2 + (j - ymid) ** 2)
            ibin = int(distance / bin)
            if ibin >= nbins:
                continue
            rad[ibin] = rad[ibin] + distance
            profile[ibin] = profile[ibin] + val
            histo[ibin] = histo[ibin] + 1

    for i in range(nbins):
        if histo[i] != 0:
            rad[i] = rad[i] / histo[i]
            profile[i] = profile[i] / histo[i]
        else:
            rad[i] = bin * (i - 0.5)
            profile[i] = np.NaN

    return rad, profile, histo


try:
    from ._cytutils import azimuthal_profile, get_radius
except ImportError:
    azimuthal_profile = slow_azimuthal_profile
    get_radius = slow_get_radius


def profile(input, origin=None, bin=1.0, nbins=None, histogram=False):
    """
    Returns axisymmetric profile of a 2d image.
    x, y[, n] = profile(image, [origin, bin, nbins, histogram])

    Parameters
    ----------
    input: `numpy.ndarray`
        2D input array
    origin: tuple of float, optional (default `None` = image center)
        Center of profile (Fits convention)
    bin: float, optional (default 1)
        Width of the profile bins (in unit of pixels).
    nbins: int, optional (default `None` = adapted on origin)
        Number of profile bins.
    histogram: bool, optional (default `False`)
        If `True`, returns the histogram.

    Returns
    -------
    radius: `numpy.ndarray`
        Profile radius in unit of pixels
    profile: `numpy.ndarray`
        Profile of input array
    [histo: `numpy.ndarray`
        Histogram of pixel count in each bin]

    Note
    ----
    This method and the underlying `azimuthal_profile` function have
    been adapted from the `pysimulators` package written by P.Chanial.

    """
    input = np.ascontiguousarray(input, np.float64)
    if origin is None:
        origin = (np.array(input.shape[::-1], np.float64) - 1) / 2
    else:
        origin = np.ascontiguousarray(origin, np.float64)

    if nbins is None:
        nbins = int(
            max(
                input.shape[0] - origin[1],
                origin[1],
                input.shape[1] - origin[0],
                origin[0],
            )
            / bin
        )

    x, y, n = azimuthal_profile(input, origin, bin, nbins)

    if histogram:
        return x, y, n
    else:
        return x, y


def integrated_profile(input, origin=None, bin=1.0, nbins=None):
    """
    Returns axisymmetric integrated profile of a 2d image.

    Parameters
    ----------
    input: numpy.ndarray
        2d input array.
    origin: tuple of int or float, optional (default `None`)
        Center of the profile. Default is the image center.
    bin: float, optional (default 1.)
        width of the profile bins (in unit of pixels).
    nbins: int, optional (default `None`)
        number of profile bins.

    Returns
    -------
    x: numpy.ndarray
        The strict upper boundary within which each integration is performed.
    y: numpy.ndarray
        The integrated profile.

    """
    x, y, n = profile(input, origin=origin, bin=bin, nbins=nbins, histogram=True)
    x = np.arange(1, y.size + 1) * bin
    y[~np.isfinite(y)] = 0
    y *= n
    return x, np.cumsum(y)


def circ_psd(image, sampling_freq=1, plot=False):
    """
    Profile of the Fourier PSD of a 2d array.

    Computes the azimuthal profile of the Fourier power spectral density
    of the image, along with the corresponding Fourier modes.
    This allows the visualization of the cutting frequency of a
    2d filter for example.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image array
    sampling_freq : float, optional (default = 1)
       The sampling frequency.
    plot : bool, optional (default `True`)
        If `True`, draws the PSD w.r.t. radius in log scale

    Returns
    -------
    radius : 1D `numpy.ndarray`
        Radius vector
    psd : 1D `numpy.ndarray`
        Circular power spectral density

    """
    assert image.ndim == 2
    assert image.shape[0] == image.shape[1]

    fimage = np.fft.fftshift(np.fft.fft2(image))
    f = np.abs(fimage) ** 2

    freq_fact = sampling_freq / image.shape[0]

    # divide by the PSD bin area = bandwidth
    f /= freq_fact**2

    f /= image.size**2

    rad, psd = profile(f)

    rad *= freq_fact

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.loglog(rad, psd, lw=1.5)
        ax.set_xlabel("sampling radius")
        ax.set_ylabel("power spectral density")
        fig.show()
    else:
        return rad, psd


def get_radius_old(value, distance, profile):
    """Return the radius corresponding to the input profile value

    Parameters
    ----------
    value: float
        Profile value which the radius is wanted
    distance: `numpy.ndarray`
        Radius grid
    profile: `numpy.ndarray`
        Image profile

    Returns
    -------
    radius: float
        Distance at which the profile matches the input value
        (same dimensionality of the distance input array)

    Remarks
    -------
    Only works for a monotonious profile

    """
    if value >= profile.max():
        radius = distance.min()
    elif value <= profile.min():
        radius = distance.max()
    else:
        wh_inside = profile <= value
        value_in = profile[wh_inside].max()
        wh_outside = profile > value
        value_out = profile[wh_outside].min()
        radius_in = distance[wh_inside][np.argmax(profile[wh_inside])]
        radius_out = distance[wh_outside][np.argmin(profile[wh_outside])]
        delta_rad = (
            (radius_out - radius_in) * (value - value_in) / (value_out - value_in)
        )
        radius = radius_in + delta_rad

    return radius
