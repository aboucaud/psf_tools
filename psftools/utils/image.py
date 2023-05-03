"""
Image submodule
---------------
Various methods that operate on 2D images.

"""
import numpy as np
from scipy import ndimage, signal

from psftools.utils.fitting import GaussianFit
from psftools.utils.profiles import circ_gauss2d

__all__ = [
    "convolve",
    "trim",
    "zero_pad",
    "mirror_pad",
    "imrotate",
    "imresample",
    "find_peak_position",
    "center_psf",
    "centroid",
    "first_moments",
    "second_moments",
    "weighted_second_moments",
    "get_e1e2",
    "center_distance",
    "circularize",
    "merge_images",
    "remove_nan",
    "interpolate_on_nan",
]


# -----
# Tools
# -----


def convolve(image, kernel, verbose=False):
    """
    Convolution in real space using scipy ndimage library

    Parameters
    ----------
    image: `numpy.ndarray`
        image to be convolved
    kernel: `numpy.ndarray`
        convolution kernel
    use_real: bool, optional
        if True, compute the convolution in real-space (default False)
    use_numpy: bool, optional
        if True, compute the Fourier convolution using numpy libraries
        (default False)
    verbose: bool, optional
        if True, print information concerning the convolution process
        (default False)

    Returns
    -------
    final_image: `numpy.ndarray`

    """
    sx0, sy0 = image.shape
    sxk, syk = kernel.shape

    # image mirroring for boundary handling
    image_pad = mirror_pad(image, sxk, syk)

    if verbose:
        print("\t", "CONVOLVE: using scipy.signal.fftconvolve method")
    # 2D convolution using FFT and scipy libraries
    # zero-padding handled by scipy
    convolved_image = signal.fftconvolve(image_pad, kernel, mode="same")

    # remove mirror padding + convolution edge
    final_image = convolved_image[sxk : sxk + sx0, syk : syk + sy0]

    return final_image


def trim(image, shape, mask=None):
    """Trim image to a given shape

    Parameters
    ----------
    image: 2D `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    mask: 2D boolean `numpy.ndarray`, optional
        Mask corresponding to the desired values in the image.
        Should have a rectangular shape given as input.
        (defaut `None`)

    Returns
    -------
    new_image: 2D `numpy.ndarray`
        Input image trimmed

    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("TRIM: null or negative shape given")

    dshape = imshape - shape
    if np.any(dshape < 0):
        raise ValueError("TRIM: target size bigger than input one")

    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise ValueError("TRIM: mask should be a boolean numpy array")
        if mask.dtype != "bool":
            raise ValueError("TRIM: mask dtype should be boolean")
        if mask.sum() != shape.prod():
            raise ValueError("TRIM: mask does not coincide with given shape")
        return image[mask].reshape(shape)

    if np.any(dshape % 2 != 0):
        raise ValueError(
            "TRIM: input and target shapes have different "
            "parity. The user should provide a mask to avoid "
            "any ambiguity"
        )

    idx, idy = np.indices(shape)
    offx, offy = dshape // 2

    return image[idx + offx, idy + offy]


def zero_pad(image, shape, position="corner", mask=False):
    """
    Extends image to a certain size with zeros

    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    mask: bool, optional
        If `True` returns a mask corresponding to the position of the
        input image in the output one. (default `False`)
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered

    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    [mask: bool `numpy.ndarray`
         The corresponding mask (if mask = True)]

    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than input one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == "center":
        if np.any(dshape % 2 != 0):
            raise ValueError(
                "ZERO_PAD: input and target shapes have different "
                "parity. The user should provide a mask to avoid "
                "any ambiguity."
            )
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    if mask:
        pad_mask = np.zeros(shape, dtype=bool)
        pad_mask[idx, idy] = 1
        return pad_img, pad_mask

    return pad_img


def mirror_pad(image, pad_x, pad_y=-1):
    """
    Extends original 2d-image on every edge mirroring its content

    Parameters
    ----------
    image: `numpy.ndarray`
        Input image
    pad_x: int
        The size of the padding in x-axis
    pad_y: int, optional
        The size of the y-axis padding if diff from x (default -1)

    Returns
    -------
    img_pad: `numpy.ndarray`
        The padded image

    """
    pad_x = int(pad_x)
    pad_y = int(pad_y)
    if pad_x < 0:
        raise ValueError("MIRROR_PAD: negative size given")
    if pad_y < 0:
        pad_y = pad_x

    # find image size
    sx0, sy0 = image.shape

    # check for appropriate size
    if pad_x > sx0 or pad_y > sy0:
        raise ValueError("Too large padding size")

    # create padded image of zeros
    img_pad = np.zeros((sx0 + 2 * pad_x, sy0 + 2 * pad_y))

    # place original image at the center of the new one
    img_pad[pad_x : pad_x + sx0, pad_y : pad_y + sy0] = image

    # top / bottom / left / right
    img_pad[:pad_x, pad_y : pad_y + sy0] = image[:pad_x][::-1]
    img_pad[-pad_x:, pad_y : pad_y + sy0] = image[-pad_x:][::-1]
    img_pad[pad_x : pad_x + sx0, :pad_y] = image[:, :pad_y][:, ::-1]
    img_pad[pad_x : pad_x + sx0, -pad_y:] = image[:, -pad_y:][:, ::-1]

    # corners
    img_pad[:pad_x, :pad_y] = image[0, 0]
    img_pad[:pad_x, -pad_y:] = image[0, -1]
    img_pad[-pad_x:, :pad_y] = image[-1, 0]
    img_pad[-pad_x:, -pad_y:] = image[-1, -1]

    return img_pad


def imrotate(image, angle, reshape=False, interp_order=1):
    """Rotate image from North to East from the given angle in degrees

    Parameters
    ----------
    image : `numpy.ndarray`
        Data array
    angle : float
        Rotation angle in degrees
    reshape : bool, optional
        Whether the input array should be contained in the output one
        (default False)
    interp_order : int, optional
        Interpolation order [1-5]
        (default 1 = linear)

    Returns
    -------
    output : `numpy.ndarray`
        Rotated data array

    """
    return ndimage.rotate(
        image, -angle, reshape=reshape, order=interp_order, prefilter=False
    )


def imresample(image, input_pscale, target_pscale, interp_order=1):
    """Resample data array from one pixel scale to another

    Parameters
    ----------
    image : `numpy.ndarray`
        Input data array
    input_pscale : float
        Pixel scale of ``image`` in arcseconds
    target_pscale : float
        Pixel scale of output array in arcseconds
    interp_order : int, optional
        Spline interpolation order [0, 5] (default 1: linear)

    Returns
    -------
    output : `numpy.ndarray`
        Resampled data array

    """
    old_size = image.shape[0]
    new_size_raw = old_size * input_pscale / target_pscale
    new_size = int(round(new_size_raw))
    ratio = new_size / old_size
    return ndimage.zoom(image, ratio, order=interp_order) / ratio**2


def psf_peak_position(image, crop_factor=5, n_iter=5):
    """
    Iterative finder of the position of the psf peak

    Parameters
    ----------
    image: `numpy.ndarray`
        Image array (PSF)
    crop_factor: float, optional
        Factor of image reduction for computational speed (default 5)
    n_iter: int, optional
        Number of maximum iterations in the smoothing process
        aimed at peak enhancement (default 5)

    Returns
    -------
    coords: tuple of ints
        Coordinates of the peak in the input image

    """
    img_size = image.shape[0]
    # Cropping the image for computing time
    # 10 < crop_size < img_size
    crop_size = min(img_size, max(img_size // crop_factor, 10))
    if crop_size % 2 == 0:
        crop_size += 1
    crop_img = trim(image, (crop_size, crop_size))
    img = crop_img.copy()

    # Retrieve number of entries with maximum value
    def n_maxima(array):
        return np.sum(array == array.max())

    # Start iterating on the flat smoothing kernel
    nit = 0
    kern_size = 3
    while (n_maxima(img) > 1) and (nit <= int(n_iter)):
        kernel = np.ones((kern_size, kern_size))
        img = ndimage.convolve(crop_img, kernel)
        kern_size += 2
        nit += 1

    if nit == n_iter:
        msg = "Peak position not found after {} smoothing level iterations"
        raise StopIteration(msg.format(n_iter))

    x_cropmax, y_cropmax = ndimage.maximum_position(img)
    offset = (img_size - crop_size) // 2

    return (x_cropmax + offset, y_cropmax + offset)


def find_peak_position(image, fit=False):
    """Basic implementation of a maximum position finder

    Parameters
    ----------
    image: `numpy.ndarray`
        Image array (PSF)
    fit: bool, optional
        Fit a Gaussian distribution to the PSF and
        return center position (default `False`)

    Returns
    -------
    coords: tuple of int
        Position of the peak as a tuple of array indices
    """
    if fit:
        fitfunc = GaussianFit(image)
        coords = fitfunc.get_center()
    else:
        coords = np.unravel_index(np.argmax(image), image.shape)

    return coords


def center_psf(image, order=1, verbose=False):
    """Shift the psf image to put the maximum at the center

    Parameters
    ----------
    image: `numpy.ndarray`
        Image array (PSF)
    order: int, optional
        interpolation order (default 1: bilinear)
    verbose: bool, optional
        If `True`, allow for printed info (default False)

    Returns
    -------
    shifted_image: `numpy.ndarray`
        Centered psf image with odd number of pixels on both axes

    """
    xsh, ysh = image.shape
    if xsh % 2 == 0:
        xsh += 1
    if ysh % 2 == 0:
        ysh += 1
    image = zero_pad(image, (xsh, ysh))
    imcenter = np.asarray(image.shape) // 2
    trucenter = np.asarray(find_peak_position(image))
    if np.all(imcenter == trucenter):
        if verbose:
            print("The array is already centered !")
        return image

    imshift = imcenter - trucenter

    if order == 1:
        prefilt = False
    else:
        prefilt = True

    return ndimage.shift(
        image, imshift, order=order, mode="constant", cval=0.0, prefilter=prefilt
    )


# def center_psf(image, crop_factor=5, conservative=True):
#     """Shift the psf image to put the maximum at the center

#     Parameters
#     ----------
#     image: `numpy.ndarray`
#         Image array (PSF)
#     crop_factor: float, optional
#         Factor of image reduction for computational speed (default 5)
#     conservative: bool, optional
#         Determines the strength of the border cleaning

#     Returns
#     -------
#     new_image:
#         Centered image with cleaned borders

#     """
#     imsize = image.shape[0]
#     center_pixel = imsize // 2

#     x_max, y_max = psf_peak_position(image, crop_factor=crop_factor)

#     shift_x = int(center_pixel - x_max)
#     shift_y = int(center_pixel - y_max)

#     if shift_x or shift_y:
#         # recenter
#         new_image = ndimage.shift(image, (shift_x, shift_y))
#         # clean borders
#         if conservative:
#             pad_x = max(abs(shift_x), abs(shift_y))
#             pad_y = max(abs(shift_x), abs(shift_y))
#         else:
#             pad_x = abs(shift_x)
#             pad_y = abs(shift_y)
#         new_image[:pad_x, :] = 0.0
#         new_image[imsize-pad_x:, :] = 0.0
#         new_image[:, :pad_y] = 0.0
#         new_image[:, imsize-pad_y:] = 0.0

#         return new_image

#     return image

# --------
# Analysis
# --------


def first_moments(input):
    """
    Compute the first moments (axis means) of the given image

    Parameters
    ----------
    input: array_like
        Input image

    Returns
    -------
    mu_x: float
        First moment on the x-axis
    mu_y: float
        First moment on the y-axis

    """
    image = np.asarray(input, dtype=float)

    # Careful of Numpy image axes indexing
    y, x = np.indices(image.shape)
    total = image.sum()
    # First moments (centroid)
    mu_x = np.sum(x * image) / total
    mu_y = np.sum(y * image) / total

    return mu_x, mu_y


def second_moments(input):
    """
    Compute the second moments of the given image

    Parameters
    ----------
    input: array_like
        Input image

    Returns
    -------
    (q_xx, q_yy, q_xy): tuple of floats
        Second moments of the image

    """
    image = np.asarray(input, dtype=float)

    # Careful of Numpy image axes indexing
    y, x = np.indices(image.shape)
    total = image.sum()
    # First moments (centroid)
    mu_x, mu_y = first_moments(image)
    # Quadrupole moments
    q_xx = np.sum((x - mu_x) * (x - mu_x) * image) / total
    q_yy = np.sum((y - mu_y) * (y - mu_y) * image) / total
    q_xy = np.sum((x - mu_x) * (y - mu_y) * image) / total

    return q_xx, q_yy, q_xy


def centroid(input, sigma, pixscl, n_iter=3):
    """
    Compute the image centroid iteratively with Gaussian weights.

    The weight is a circular two-dimensional Gaussian function centered
    on the first moments of the image at each iteration if the center is
    not specified as input.

    Parameters
    ----------
    input: array_like
        Input image
    sigma: float
        Width of the weighting Gaussian in arcsec
    pixscl: float
        Pixel scale of the image to scale the Gaussian function
    n_iter: int, optional
        Number of iterations of the weighting (default 3)

    Returns
    -------
    mu_x, mu_y
        Coordinates of the cendroid in the image plane

    """
    image = np.asarray(input, dtype=float)

    # Scale the Gaussian width to the image pixel scale
    true_sigma = sigma / pixscl

    if n_iter <= 0:
        return first_moments(image)

    mu_x, mu_y = first_moments(image)
    for _ in range(n_iter):
        gweight = circ_gauss2d(image.shape, mu_x, mu_y, true_sigma)
        imweighted = image.copy() * gweight
        mu_x, mu_y = first_moments(imweighted)

    return mu_x, mu_y


def weighted_second_moments(input, sigma, pixscl, n_iter=3, center=None):
    """
    Compute iteratively the weighted second moments of the image.

    The weight is a circular two-dimensional Gaussian function centered
    on the first moments of the image at each iteration if the center is
    not specified as input.

    Parameters
    ----------
    input: array_like
        Input image
    sigma: float
        Width of the weighting Gaussian in arcsec
    pixscl: float
        Pixel scale of the image to scale the Gaussian function
    n_iter: int, optional
        Number of iterations of the weighting (default 3)
    center: tuple of floats, optional
        Fixed center for the Gaussian weight (default `None`)

    Returns
    -------
    Q_xx, Q_yy, Q_xy: tuple of floats
        Second moments (quadrupoles) of the image

    """
    image = np.asarray(input, dtype=float)

    if n_iter <= 0:
        return second_moments(image)

    if center is None:
        mu_x, mu_y = centroid(image, sigma, pixscl, n_iter=n_iter)
    else:
        try:
            mu_x, mu_y = center
        except ValueError:
            print("Wrong input for center coordinates, " "switching to first moments")
            mu_x, mu_y = first_moments(image)
    gweight = circ_gauss2d(image.shape, mu_x, mu_y, sigma / pixscl)
    imweighted = image.copy() * gweight

    return second_moments(imweighted)


def get_e1e2(input, sigma=0.75, pixscl=0.1, n_iter=3):
    """
    Compute the ellipticity components of the given image
    as the weighted second moments of the image.

    Parameters
    ----------
    input: array_like
        Input image
    sigma: float, optional
        Width of the weighting Gaussian in arcsec (default 0.75)
    pixscl: float, optional
        Pixel scale of the image (default 0.1)
    n_iter: int, optional
        Number of iterations of the weighting (default 3)

    Returns
    -------
    e1: float
        Real component of the ellipticity
    e2: float
        Imaginary component of the ellipticity

    """
    q_xx, q_yy, q_xy = weighted_second_moments(input, sigma, pixscl, n_iter=n_iter)

    # Ellipticity components
    e1 = (q_xx - q_yy) / (q_xx + q_yy)
    e2 = 2 * q_xy / (q_xx + q_yy)

    return e1, e2


def get_ellipticity(input, sigma=0.75, pixscl=0.1, n_iter=3):
    """
    Compute the ellipticity the given image

    Parameters
    ----------
    input: array_like
        Input image
    sigma: float, optional
        Width of the weighting Gaussian in arcsec (default 0.75)
    pixscl: float, optional
        Pixel scale of the image (default 0.1)
    n_iter: int, optional
        Number of iterations of the weighting (default 3)

    Returns
    -------
    ellipticity: float
        Ellipticity comprised between 0 and 1

    """
    e1, e2 = get_e1e2(input, sigma=sigma, pixscl=pixscl, n_iter=n_iter)

    ellipticity = np.sqrt(e1**2 + e2**2)
    if ellipticity > 1:
        print("Warning: derived ellipticity greater than one")
        return 1.0

    return ellipticity


def get_major_minor_axes(input, sigma=0.75, pixscl=0.1, n_iter=3):
    """
    Compute the major and minor axes of the given image
    from the quadrupole moments

    Parameters
    ----------
    input: array_like
        Input image
    sigma: float, optional
        Width of the weighting Gaussian in arcsec (default 0.75)
    pixscl: float, optional
        Pixel scale of the image (default 0.1)
    n_iter: int, optional
        Number of iterations of the weighting (default 3)

    Returns
    -------
    a, b: tuple of floats
        Major and minor axes of the image

    """
    q_xx, q_yy, q_xy = weighted_second_moments(input, sigma, pixscl, n_iter=n_iter)

    a = np.sqrt(0.5 * (q_xx + q_yy + np.sqrt((q_xx - q_yy) ** 2 + 4 * q_xy**2)))
    b = np.sqrt(0.5 * (q_xx + q_yy - np.sqrt((q_xx - q_yy) ** 2 + 4 * q_xy**2)))

    # a should be the major axis
    if a < b:
        a, b = b, a

    return a, b


def get_r_squared(input, sigma=0.75, pixscl=0.1, n_iter=3):
    """
    Compute R2 factor in arcseconds

    Parameters
    ----------
    input: array_like
        Input image
    sigma: float, optional
        Width of the weighting Gaussian in arcsec (default 0.75)
    pixscl: float, optional
        Pixel scale of the image (default 0.1)
    n_iter: int, optional
        Number of iterations of the weighting (default 3)

    Returns
    -------
    r_squared: float
        R2 factor in arcseconds

    """
    q_xx, q_yy, _ = weighted_second_moments(input, sigma, pixscl, n_iter=n_iter)

    r_squared_pix = q_xx + q_yy

    return r_squared_pix * pixscl**2


def center_distance(size_x, size_y=0):
    """Return the pixel distance to the array center

    Parameters
    ----------
    size_x: int
        Size along the x-axis of the square array
    size_y: int, optional
        Size along the y-axis if different from x
        (default 0)

    Returns
    -------
    distance: 2d `numpy.ndarray`
        Euclidian distance (in pixels) from the center of the array

    """
    size_x = int(size_x)
    size_y = int(size_y)
    assert size_x >= 0, "The size must be positive"
    if size_y <= 0:
        size_y = size_x
    assert size_x % 2 != 0, "The size must be odd"
    assert size_y % 2 != 0, "The size must be odd"

    array_x, array_y = np.mgrid[:size_x, :size_y]
    array_x -= size_x // 2
    array_y -= size_y // 2
    distance = np.sqrt(array_x**2 + array_y**2)

    return distance


# ------
# Aniano
# ------


def circularize(image, n_iter=14, interp_order=2, log=False):
    """
    Circularize an image.

    Successively rotate the image and add it to itself
    with angles decreasing by a factor 2 every iteration.
    This is numerically equivalent to computing the average
    over 2^n rotations where n is the number of iterations.
    THe default choice of n_iter = 14 hence produces an image
    invariant under rotations of any angle that is a multiple
    of 360/2^14 = 360/16384 ~ 0.022 deg

    Parameters
    ----------
    image: real `numpy.ndarray`
        Image to circularize
    n_iter: int, optional
        Number of iterations in the process (default 14)
    interp_order: int, optional
        Spline interpolation order in the rotation
            * 0 : linear
            * 1 : bilinear
            * 2 : bicubic (default)

    Returns
    -------
    Circularized image : real `numpy.ndarray`

    """
    if log:
        assert np.all(image != 0), "Cannot use log if zeros in image"
        image = np.log(image)

    angle_gen = (360 / 2 ** (i + 1) for i in range(n_iter)[::-1])
    for angle in angle_gen:
        rotated_image = ndimage.rotate(
            image, angle, reshape=False, order=interp_order, prefilter=False
        )
        image = 0.5 * (image + rotated_image)

    if log:
        return np.exp(image)

    return image


def merge_images(core, tail, radius_transition):
    """Merge a core image with an analytic tail

    Parameters
    ----------
    core: `numpy.ndarray`
        Core image
    tail: `numpy.ndarray`
        Analytic tail
    transition_radius: float
        Characteristic radius where the transition
        between core and tail occurs

    Returns
    -------
    merged_image: `numpy.ndarray`
        The merged image

    """
    if core.shape != tail.shape:
        # reshape(tail)
        pass

    distance = center_distance(core.shape[0])
    cut_f_lo = radius_transition * 0.95
    cut_f_hi = radius_transition * 1.05

    filt = 0.5 * (1.0 + np.cos(np.pi * (distance - cut_f_lo) / (cut_f_hi - cut_f_lo)))
    filt[distance < cut_f_lo] = 1.0
    filt[distance >= cut_f_hi] = 0.0

    merged_image = core * filt + tail * (1 - filt)

    percent_in_core = core.sum() / merged_image.sum() * 100
    print(
        "%d percent of the PSF flux is in the added analytic wings."
        % (100 - percent_in_core)
    )

    return merged_image


# ----
# NaNs
# ----


def remove_nan(image):
    """Replaces NaNs by zeros in the given array"""
    return np.nan_to_num(image)


def interpolate_on_nan(image):
    """Replaces NaNs by interpolated values on the given array"""
    mask = np.isnan(image)
    image[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), image[~mask])
    return image
