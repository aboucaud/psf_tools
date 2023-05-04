#!/usr/bin/env python
"""
# make_kernel

Compute the transition kernel between two PSFs

It takes two images (FITS files) as input, a high-frequencies

The computation of the kernel uses deconvolution with a Wiener filter
where the high-frequencies of the image (noise) are penalized by a
Laplacian filter, at an amplitude given by the input regularization
parameter.

Alternatively, the user can decide to use the Aniano recipe
(circularization + high-frequency filtering) by passing the
`--aniano` keyword.

## Usage

```
make_kernel [-h] [psf_input] [psf_target] [-o [output]]
            [-p [pixel_scale]] [--angle_input] [--angle_target]
            [-R [regfact]] [-A, --aniano] [-m, --minimize]
            [-v, --verbose]
```

## Arguments

* `psf_input`:        path to the input PSF (FITS image)
* `psf_target`:       path to the low resolution PSF (FITS image)
* `-o output`:        the output filename and path
* `-p pixel_scale`:   pixel scale of the output fits in arcsec

## Optionals

* `-h`, `--help`:     print help
* `--angle_input`:    rotate image from North to East given the angle
                      in degrees (default 0.0)
* `--angle_target`:   rotate image from North to East given the angle
                      in degrees (default 0.0)
* `-R`, `--regfact`:  regularization factor
                      (default 1e-5)
* `-A`, `--aniano`:   use the Aniano method instead of Wiener filtering
                      (default False)
* `-m`, `--minimize`: minimize the size of the output kernel
                      (default False) (currently unavailable)
* `-v`, `--verbose`:  print information while running the script
                      (default False)

"""

import os
from os.path import basename

import numpy as np

import psftools.utils as utils
from psftools import PSF

OUTPUTDIR = os.getcwd()


def print_help(*msg):
    print("make_kernel:", *msg)


def prepare_input_psfs(args):
    """
    Compute preliminary steps for kernel computation

    Parameters
    ----------
    args: `argparse.Namespace`
        Namespace with input information stored

    Returns
    -------
    psf_input: `psftools.PSF`
        PSF object for the high res PSF
    psf_target: `psftools.PSF`
        PSF object for the low res PSF

    """
    psf_input = PSF(args.psf_input, backup=False, verbose=args.verbose)
    psf_target = PSF(args.psf_target, backup=False, verbose=args.verbose)
    if args.verbose:
        print_help("Images loaded")

    if args.pixel_scale == 0.0:
        args.pixel_scale = min(psf_input.pixel_scale, psf_target.pixel_scale)
        if args.verbose:
            print_help(
                "Output pixel scale not specified.",
                "Set to the minimum of both input ones.",
            )

    # rotate the images (if indicated)
    if args.angle_input != 0.0:
        psf_input.rotate(args.angle_input)
        if args.verbose:
            print_help(
                f"Input PSF rotated {args.angle_input:.2f} degrees from North to East"
            )
    if args.angle_target != 0.0:
        psf_target.rotate(args.angle_target)
        if args.verbose:
            print_help(
                f"Target PSF rotated {args.angle_target:.2f} degrees from North to East"
            )

    # Normalize
    psf_input.normalize()
    psf_target.normalize()

    # resample
    for psf in [psf_input, psf_target]:
        psf.make_odd_square()
        # resample if necessary
        if psf.pixel_scale != args.pixel_scale:
            psf.resample(args.pixel_scale)
            if args.verbose:
                print_help(f"{psf.filename} resampled")

    return psf_input, psf_target


def prepare_images_aniano(psf_input, psf_target):
    # Find common size
    size = max(psf_input.shape[0], psf_target.shape[0])

    # circularization
    for psf in [psf_input, psf_target]:
        if psf.shape[0] < size:
            psf.resize(size)
        psf.normalize(tomax=True)
        psf.circularize()
        psf.compute_fft()
        psf.circularize_fft()
        psf.filter_fft()

    return psf_input, psf_target


def compute_circular_kernel(psf_input, psf_target):
    """
    Compute the kernel using circularization

    Parameters
    ----------
    psf_input: `psftools.PSF`
        PSF object for the high res PSF
    psf_target: `psftools.PSF`
        PSF object for the low res PSF

    Returns
    -------
    kernel: `numpy.ndarray`
        Output circular kernel
    header: fits.Header
        Kernel header freshly formatted

    """
    psf_input, psf_kernel = prepare_images_aniano(psf_input, psf_target)
    # Inverse the high res PSF Foutrier transform and filter it
    imfft_input_inv = psf_input.get_inverse_fft_image()
    # Take the "to" PSF FT
    imfft_target = psf_target.image_fft
    # Form the kernel by convolution in Fourier space (product)
    psf_fft = imfft_input_inv * imfft_target

    # Inverse Fourier transform and shift to recover the real kernel
    psf_shifted = np.abs(np.fft.ifft2(psf_fft))
    raw_kernel = np.fft.ifftshift(psf_shifted)

    kernel = utils.circularize(raw_kernel)
    kernel /= kernel.max()
    kernel[kernel < 1.0e-8] = 0.0

    dist = utils.center_distance(kernel.shape[0])
    kernel[dist > kernel.shape[0] // 2] = 0.0

    return kernel


def compute_wiener_kernel(psf_input, psf_target, reg_fact):
    """
    Compute the kernel using Wiener filtering

    Parameters
    ----------
    psf_input: `psftools.PSF`
        PSF object for the high res PSF
    psf_target: `psftools.PSF`
        PSF object for the low res PSF
    reg_fact: float
        Regularization factor for the inversion

    Returns
    -------
    kernel: `numpy.ndarray`
        Output circular kernel
    header: fits.Header
        Kernel header freshly formatted

    Notes
    -----
    Solves the system y = H(x) + n
    where x is the desired kernel and n is the noise
    using a Wiener filtering method with regularization
    and a high-pass penalization (Laplacian)

    """
    kernel = utils.wiener_laplace(psf_target.image, psf_input.image, reg_fact=reg_fact)

    return kernel


def compute_bayesian_kernel(psf_input, psf_target, sigma=1, reg_fact=0.0005):
    """
    Compute the kernel using bayesian methods

    Parameters
    ----------
    psf_input: `psftools.PSF`
        PSF object for the high res PSF
    psf_target: `psftools.PSF`
        PSF object for the low res PSF

    Returns
    -------
    kernel: `numpy.ndarray`
        Output circular kernel
    header: fits.Header
        Kernel header freshly formatted

    Notes
    -----
    Solves the system y = H(x) + n
    where x is the desired kernel and n is the noise
    using preconditioned conjugate gradient

    """
    kernel = utils.pcg_solver_single(
        psf_target.image, psf_input.image, sigma, reg_fact=reg_fact
    )

    kernel /= kernel.max()
    kernel[kernel < 1.0e-8] = 0.0

    return kernel


def init_fits_and_header(kernel, args):
    """
    Format the output kernel header along with FITS data

    Parameters
    ----------
    kernel: `numpy.ndarray`
        The real space convolution kernel

    Returns
    -------
    header: `fits.PrimaryHDU`
        Output kernel formatted FITS

    """
    kerfits = utils.create_fits(kernel)
    hdr = kerfits.header

    hdr.add_comment("=" * 50)
    hdr.add_comment("")
    hdr.add_comment("File written with make_kernel.py and the PSF tools package")
    hdr.add_comment("")
    hdr.add_comment(
        f"Kernel from PSF {basename(args.psf_input)} to {basename(args.psf_target)}"
    )
    if not args.aniano:
        hdr.add_comment(f"using a regularisation parameter R = {args.regfact:1.1e}")
    hdr.add_comment("")
    hdr.add_comment("=" * 50)

    hdr["CD1_1"] = (args.pixel_scale, "pixel scale in deg.")
    hdr["CD1_2"] = (0, "pixel scale in deg.")
    hdr["CD2_1"] = (0, "pixel scale in deg.")
    hdr["CD2_2"] = (args.pixel_scale, "pixel scale in deg.")

    return kerfits


def minimize_psf_size(kernel, header):
    """
    Trim the kernel to keep an array containing 99.9% of its energy

    Parameters
    ----------
    kernel: `numpy.ndarray`
        The real space convolution kernel

    Returns
    -------
    kernel_reduced: `numpy.ndarray`
        A trimmed version of the kernel

    """
    ana_dict = utils.analyze_radial_profile(kernel)
    pixradius = ana_dict["radius"]
    integ_profile = ana_dict["integrated_profile"]
    pixmax = np.ceil(
        utils.get_radius(0.999, pixradius, integ_profile / integ_profile[-1])
    )
    new_size = 2 * pixmax + 1

    kernel_reduced = utils.trim(kernel, (new_size, new_size))
    dist = utils.center_distance(new_size)
    kernel_reduced[dist > new_size] = 0.0

    # Update header info
    x0, y0 = kernel.shape
    header["NAXIS1"] = y0
    header["NAXIS2"] = x0
    header["CRPIX1"] = y0 // 2
    header["CRPIX2"] = x0 // 2

    return kernel_reduced, header


def parse_args():
    """Command-line parser"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the transition kernel between two PSFs"
    )

    parser.add_argument(
        "psf_input",
        nargs="?",
        metavar="psf_input",
        help="the kernel with highest resolution",
    )

    parser.add_argument(
        "psf_target",
        nargs="?",
        metavar="psf_target",
        help="the kernel with lowest resolution",
    )

    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        metavar="output",
        default="",
        const="",
        dest="output",
        help="the output file name",
    )

    parser.add_argument(
        "-p",
        "--pix",
        nargs="?",
        metavar="pixel_scale",
        type=float,
        default=0.0,
        const=0.0,
        dest="pixel_scale",
        help="pixel scale of the output fits in arcsec",
    )

    parser.add_argument(
        "--angle_input",
        nargs="?",
        metavar="angle_input",
        type=float,
        default=0.0,
        const=0.0,
        dest="angle_input",
        help="rotation angle in degrees to apply to `psf_input`",
    )

    parser.add_argument(
        "--angle_target",
        nargs="?",
        metavar="angle_target",
        type=float,
        default=0.0,
        const=0.0,
        dest="angle_target",
        help="rotation angle in degrees to apply to `psf_target`",
    )

    parser.add_argument(
        "-R",
        "--regfact",
        nargs="?",
        metavar="regfact",
        type=float,
        default=1.0e-5,
        const=1.0e-5,
        dest="regfact",
        help="regularization factor for the Wiener filtering",
    )

    parser.add_argument(
        "-A",
        "--aniano",
        action="store_true",
        help="use the Aniano method (circularization + filtering)",
    )

    # parser.add_argument(
    #     '-m',
    #     '--minimize',
    #     action='store_true',
    #     help="minimize the size of the output kernel")

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print information while running the script",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.psf_input or not args.psf_target:
        raise OSError("Missing input filename")
    if not args.output:
        raise OSError("Missing output filename")

    psf_input, psf_target = prepare_input_psfs(args)

    if args.aniano:
        kernel = compute_circular_kernel(psf_input, psf_target)
    else:
        kernel = compute_wiener_kernel(psf_input, psf_target, reg_fact=args.regfact)
        # kernel = compute_bayesian_kernel(psf_input, psf_target,
        #                                  reg_fact=args.regfact)
    if args.verbose:
        print_help("Transformation kernel computed")

    # Create FITS object with appropriate header
    kerfits = init_fits_and_header(kernel, args)

    # # Minimize kernel size if desired
    # if args.minimize:
    #     kernel, header = minimize_psf_size(kernel, header)
    #     minim_output = '{}-minimal{}'.format(*os.path.splitext(args.output))
    #     utils.write_fits(minim_output, kernel, header)
    #     if args.verbose:
    #         print_help("Transformation kernel size reduction completed ",
    #                   "down to 99.9% of its total power")

    # Write FITS file on disk, don't overwrite if already exists
    kerfits.writeto(args.output, clobber=False)
    if args.verbose:
        print_help(f"Output kernel saved in {args.output}")


if __name__ == "__main__":
    main()
