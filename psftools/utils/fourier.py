"""
Fourier submodule
-----------------
Methods involving Fourier transform or the Fourier domain.
+ Wrapper methods for pyFFTW

"""
from functools import partial

import numpy as np

from psftools.utils.image import zero_pad

__all__ = ["fft2", "ifft2", "ufft2", "uifft2", "psf2otf"]

try:
    import pyfftw

    def pyfftw_builder(image, method):
        """
        Compute the Fourier transform of an image using multithreading

        Parameters
        ----------
        image: real `numpy.ndarray`
            Input image
        method: str
            pyFFTW method

        Returns
        -------
        fft_image: complex `numpy.ndarray`
            Fourier transform of input image

        """
        # Optimal alignment in bytes for the CPU architecture
        opt_n = pyfftw.simd_alignment
        # Check
        if not pyfftw.is_n_byte_aligned(image, opt_n):
            image = pyfftw.n_byte_align(image, opt_n, dtype=image.dtype)
        # FFTW wrapper
        fft_method = getattr(pyfftw.builders, method)
        fft_obj = fft_method(image)

        return fft_obj()

    fft2 = partial(pyfftw_builder, method="fft2")
    ifft2 = partial(pyfftw_builder, method="ifft2")
except ImportError:
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2

# def fft2(image):
#     """
#     Compute the Fourier transform of an image using multithreading

#     Parameters
#     ----------
#     image: real `numpy.ndarray`
#         Input image

#     Returns
#     -------
#     fft_image: complex `numpy.ndarray`
#         Fourier transform of input image

#     """
#     # Optimal alignment in bytes for the CPU architecture
#     opt_n = pyfftw.simd_alignment
#     # Check
#     if not pyfftw.is_n_byte_aligned(image, opt_n):
#         image = pyfftw.n_byte_align(image, opt_n)
#     # FFTW wrapper
#     fftobj = pyfftw.builders.fft2(image)

#     return fftobj()


# def ifft2(fft_image):
#     """
#     Compute the inverse Fourier transform of an image using multithreading

#     Parameters
#     ----------
#     fft_image: complex `numpy.ndarray`
#         Input image (Fourier domain)

#     Returns
#     -------
#     image: complex `numpy.ndarray`
#         Inverse Fourier transform of input image

#     """
#     # Optimal alignment in bytes for the CPU architecture
#     opt_n = pyfftw.simd_alignment
#     # Check
#     if not pyfftw.is_n_byte_aligned(fft_image, opt_n):
#         fft_image = pyfftw.n_byte_align(fft_image, opt_n)
#     # FFTW wrapper
#     fftobj = pyfftw.builders.ifft2(fft_image)

#     return fftobj()


def ufft2(image):
    """Unitary fft2"""
    norm = np.sqrt(image.size)
    return fft2(image) / norm


def uifft2(image):
    """Unitary ifft2"""
    norm = np.sqrt(image.size)
    return ifft2(image) * norm


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.

    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.

    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in `shape`, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.

    Parameters
    ----------
    psf : `numpy.ndarray`
        Input PSF array
    shape : tuple of int
        Output shape of the OTF array

    Returns
    -------
    otf : `numpy.ndarray`
        Output OTF array

    Notes
    -----
    Adapted from MATLAB psf2otf function

    """
    if np.all(psf == 0):
        otf = np.zeros_like(psf)
    else:
        inshape = psf.shape
        # Pad the PSF to shape
        # if inshape[0] < shape:
        psf = zero_pad(psf, shape, position="corner")

        # Circularly shift OTF so that the 'center' of the PSF is
        # [0,0] element of the array
        for axis, axis_size in enumerate(inshape):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)

        # Compute the OTF
        otf = fft2(psf)

        # Estimate the rough number of operations involved in the FFT
        # and discard the PSF imaginary part if within roundoff error
        # roundoff error  = machine epsilon = sys.float_info.epsilon
        # or np.finfo().eps
        n_ops = np.sum(psf.size * np.log2(psf.shape))
        otf = np.real_if_close(otf, tol=n_ops)

    return otf
