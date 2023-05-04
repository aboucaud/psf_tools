#!/usr/bin/env python
"""
=============
homogenize.py
=============
Perform a convolution of a FITS image with an homogenization kernel

Usage:
    homogenize.py [-h] [-v] [image] [-k [kernel]] [-o [outputfile]]
                  [--real] [--numpy]
Args:
    image:      path to FITS image to be homogenized
Optionals:
    -h, --help: print help
    -v:         verbose-mode
    -k kernel:  path to the convolution kernel
    -o output:  provides the name of output FITS file
    --real:     uses real-space convolution instead of Fourier-space
    --numpy:    uses numpy methods instead of scipy's (for Fourier conv)

"""
import os

from psftools import PSF, ImageFits
from psftools.utils.fits import write_fits
from psftools.utils.image import zero_pad


def print_help(*msg):
    print("homogenize.py:", *msg)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convolve an image using a homogenization kernel"
    )

    parser.add_argument(
        "imgfile", nargs="?", metavar="image", help="image to be homogenized"
    )

    parser.add_argument(
        "kernfile",
        nargs="?",
        metavar="kernel",
        help="kernel to convolve with",
    )

    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        metavar="outputfile",
        default="output.fits",
        const="output.fits",
        dest="outfile",
        help="the output file name",
    )

    parser.add_argument(
        "--real",
        dest="real",
        action="store_true",
        help="compute the convolution in real space",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print information while running the script",
    )

    parser.add_argument(
        "-f", "--force", action="store_true", help="force file overwritting"
    )

    args = parser.parse_args()

    # check output filename
    filepath = args.outfile
    if not args.force and os.path.isfile(filepath):
        print(f"{filepath} already existing, skipping..")
        # i += 1
        # filepath = "{0}-{1}.fits".format(args.outfile, i)
    else:
        # Loading data
        # ------------

        # image
        obj_image = ImageFits(args.imgfile, backup=False, verbose=args.verbose)
        if args.verbose:
            print_help("Image file loaded")

        # kernel
        obj_kernel = PSF(args.kernfile, backup=False, verbose=args.verbose)
        if args.verbose:
            print_help("Kernel file loaded.", "Preparing the images...")

        # Preparation
        # -----------
        img_size_x, img_size_y = obj_image.shape
        pixel_scale_img = obj_image.pixel_scale

        # zero padding HARD CODED !!!!!!!!
        # padding_arcsec = 100
        padding_arcsec = 0.3 * pixel_scale_img * max(img_size_x, img_size_y)
        # padding_factor = 1.3
        pixels_added = padding_arcsec // pixel_scale_img
        img_size_x_new = img_size_x + 2 * pixels_added
        img_size_y_new = img_size_y + 2 * pixels_added
        # img_size_x_new = int(padding_factor * img_size_x)
        # img_size_y_new = int(padding_factor * img_size_y)
        # pixel_added =

        img = zero_pad(
            obj_image.image, (img_size_x_new, img_size_y_new), position="center"
        )
        if args.verbose:
            print_help("Image padded")

        # adjust convolution kernel  to the image resolution
        pixel_scale_kern = obj_kernel.pixel_scale
        if not obj_kernel.is_square:
            obj_kernel.make_odd_square()

        if (pixel_scale_kern / pixel_scale_img - 1) > 0.05:
            if args.verbose:
                print_help(
                    "The convolution kernel and the image are in grids",
                    "of different pixel size.",
                )
                print_help("Transforming kernel into the correct pixel size")

            obj_kernel.resample(pixel_scale_img)

        max_kern_size = min(img_size_x, img_size_y)
        if max_kern_size % 2 == 0:
            max_kern_size -= 1
        kern_size = obj_kernel.shape[0]
        if kern_size > max_kern_size:
            if args.verbose:
                print_help("Resizing the kernel")

            obj_kernel.resize(max_kern_size)

        if not obj_kernel.is_centered:
            obj_kernel.center_psf()

        obj_kernel.normalize()

        # Convolution
        # -----------
        if args.real:
            from scipy.ndimage import convolve

            convolved_image = convolve(img, obj_kernel.image, mode="constant", cval=0.0)
        else:
            from scipy.signal import fftconvolve

            convolved_image = fftconvolve(img, obj_kernel.image, mode="same")

        result_image = convolved_image[
            pixels_added : pixels_added + img_size_x,
            pixels_added : pixels_added + img_size_y,
        ]

        if args.verbose:
            print_help("Convolution completed, writing FITS file...")

        # write output image
        write_fits(filepath, result_image, obj_image.header)

        if args.verbose:
            print_help(f"Output image written in {filepath}")


if __name__ == "__main__":
    main()
