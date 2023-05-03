"""
Deconvolution module
--------------------
Methods used for image deconvolution

"""
import numpy as np

from psftools.utils.fourier import psf2otf, ufft2, uifft2

try:
    import pyoperators
except ImportError:
    print(
        "The pyoperator library is not installed.",
        "Cannot use the pcg_solver_single method.",
    )

LAPLACIAN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

__all__ = ["pcg_solver_single", "wiener_reg", "wiener_laplace"]


def pcg_solver_single(image, kernel, sigma, reg_parm=0.0005):
    """
    Solve equation A x = b using preconditioned conjugate gradient algorithm

    Parameters
    ----------
    image: `numpy.array`
    kernel: `numpy.array`
    sigma: int or float
    reg_parm: float, optional
        Regularisation parameter for the inversion method

    Returns
    -------
    deconvolved_image: `numpy.array`
        The deconvolved image

    """
    H = pyoperators.ConvolutionOperator(kernel, kernel.shape)
    if not sigma:
        A = H.T * H + reg_parm * pyoperators.I
        solution = pyoperators.pcg(A, H.T(image))
    else:
        invN = pyoperators.BlockDiagonalOperator(1 / sigma**2, new_axisin=0)
        A = H.T * invN * H + reg_parm * pyoperators.I
        solution = pyoperators.pcg(A, H.T(invN(image)))

    if not solution["success"]:
        print(solution["message"])

    return solution["x"]


def wiener_reg(image, psf, reg_fact, clip=False, clipfact=1):
    """Deconvolution using a Wiener filter

    Parameters
    ----------
    image: `numpy.ndarray`
        2D input array
    psf: `numpy.ndarray`
        2D kernel array
    reg_fact: float
        Regularisation parameter for the inversion method
    clip: bool, optional
        If `True`, enforces the non-amplification of the noise
        (default `False`)
    clipfact: int, optional
        Level of the amplitude clipping (default 1)

    Returns
    -------
    deconv_image: `numpy.ndarray`
        2D deconvolved image

    """
    # Make sure to get an optical transfer function (OFT)
    # for one of the image (chosen to be the convolution kernel)
    # as it will preserve the phase during the convolution process.
    fs = psf2otf(psf, image.shape)
    fd = ufft2(image)

    filt = fs.conj() / (np.abs(fs) ** 2 + reg_fact * np.abs(fs[0, 0]) ** 2)

    ftk = fd * filt

    clip_value = np.real(fd[0, 0] / fs[0, 0] * clipfact)
    ftk = fft_clipping(ftk, clip_value)

    kernel = uifft2(ftk)
    n_ops = np.sum(kernel.size * np.log2(kernel.shape))

    return np.real_if_close(kernel, tol=n_ops)


def fft_clipping(fimage, value, side="right"):
    # Store phase and amplitude
    amp = np.abs(fimage)
    phase = np.angle(fimage)
    if side == "right":
        # Make sure you do not amplify any high frequencies
        amp.clip(amp.min(), value)
    elif side == "left":
        amp.clip(value, amp.max())
    else:
        print("Unknown parameter for clipside")
    # Recover phase
    return amp * np.exp(1j * phase)


def wiener_laplace(image, psf, reg_fact, clip=True):
    """Deconvolution using a Wiener filter and high-freq penalization

    The signal is penalized by a 2D Laplacian operator that serves as
    a high-pass filter for the regularization process.

    Parameters
    ----------
    image: `numpy.ndarray`
        2D input array
    psf: `numpy.ndarray`
        2D kernel array
    reg_fact: float
        Regularisation parameter for the inversion method
    clip: bool, optional
        If `True`, enforces the non-amplification of the noise
        (default `True`)

    Returns
    -------
    deconv_image: `numpy.ndarray`
        2D deconvolved image

    """
    reg = psf2otf(LAPLACIAN, image.shape)

    trans_func = psf2otf(psf, image.shape)

    wiener_filter = np.conj(trans_func) / (
        np.abs(trans_func) ** 2 + reg_fact * np.abs(reg) ** 2
    )

    kernel = uifft2(wiener_filter * ufft2(image))

    if clip:
        kernel.clip(-1, 1)

    n_ops = np.sum(kernel.size * np.log2(kernel.shape))

    return np.real_if_close(kernel, tol=n_ops)
