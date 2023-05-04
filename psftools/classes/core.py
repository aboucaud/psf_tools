"""
Core submodule
--------------

"""
import os

import numpy as np

import psftools.utils as utils
from psftools.utils import print_help

__all__ = ["ImageFits"]

THRESH_VAL = 1.0e-8


class ImageFits:
    """Base class for handling FITS images

    Parameters
    ----------
    input : str, list or numpy.ndarray
        Path to the image FITS file or image data
    pixel_scale : float, optional (default 0.0)
        Pixel scale of input image data in arcseconds
    backup : bool, optional (default `False`)
        Create a backup when loading the data
    verbose : bool, optional (default `False`)
        Add some verbosity to the actions

    Attributes
    ---------
    filename : str
        Path to the image FITS file
    image : array_like
        Image data cube
    header : `fits.Header`
        Image header
    image_fft : array_like
        Array of the Fourier transform of the images
    verbose: bool
        If `True`, print debug messages
    _image_copy : array_like, optional
        Backup image
    _header_copy : `fits.Header`, optional
        Backup header

    """

    def __init__(self, input, pixel_scale=0.0, backup=False, verbose=False):
        # Input data
        self.filename = input
        # Main arguments
        self.image = []
        self.header = []
        # Fourier transform
        self.image_fft = []
        # Verbose keyword
        self.verbose = verbose
        # Backup copies
        self._image_copy = None
        self._header_copy = None
        # Circularization boolean
        self._circularized = False

        # Load data and header
        self.load_data(pixel_scale=pixel_scale, backup=backup)

    def load_data(self, pixel_scale, backup):
        """Load kernel and header from external file and store a copy

        Parameters
        ----------
        pixel_scale : float
            Pixel scale of input image data in arcseconds
        backup : bool
            Create a backup when loading the data

        """
        if isinstance(self.filename, str):
            extension = os.path.splitext(self.filename)[1]
            if extension.lower() == ".fits":
                img, hdr = utils.load_fits_with_header(self.filename)
            else:
                raise OSError(f"{extension} file extension not supported")
        else:
            if pixel_scale == 0.0:
                raise OSError(
                    "The pixel scale should be specified when creating a "
                    "{} class from an array.".format(self.__class__.__name__)
                )
            if isinstance(self.filename, list):
                img = np.asarray(self.filename, dtype=float)
            elif isinstance(self.filename, np.ndarray):
                img = self.filename
            hdr = utils.create_header(img)
            hdr["PIXSCALE"] = pixel_scale

        if img.ndim < 2 or img.ndim > 3:
            raise ValueError(
                f"The provided data has {img.ndim} dimension(s). "
                + "It should either be an image (2D) or a data cube (3D)."
            )

        if img.size == 0:
            raise ValueError("The provided image is empty")

        self.image = img
        self.header = hdr

        self._format_header()

        if backup:
            self.make_copy()

        if self.verbose:
            print_help("load_data", f"File {self.filename} loaded.")

    @property
    def image(self):
        """The current image"""
        return self._image

    @image.setter
    def image(self, image):
        """Replace current image array by input array

        Parameters
        ----------
        image : `numpy.ndarray`
            Image data

        """
        self._image = image

    @property
    def header(self):
        """Return current kernel"""
        return self._header

    @header.setter
    def header(self, header):
        """Replace current kernel header by input header

        Parameters
        ----------
        header : fits.Header
            Header file corresponding to the image data

        """
        self._header = header

    def update_header_key(self, key, value):
        """Update of add a given key of the image header"""
        if self.verbose:
            if key in self.header.keys():
                msg = f"value updated for {key}"
            else:
                msg = f"value added for {key}"
            print_help("header", msg)

        self.header[key] = value

    def update_header_comment(self, key, comment):
        """Update of add a given key of the image header"""
        if self.verbose:
            if key in self.header.keys():
                msg = f"comment updated for {key}"
            else:
                msg = f"comment added for {key}"
            print_help("header", msg)

        self.header.comments[key] = comment

    def delete_header_key(self, key):
        """Delete a given key of the image header"""
        if key in self.header.keys():
            del self.header[key]
            if self.verbose:
                msg = f"entry {key} deleted"
                print_help("header", msg)

    def _format_header(self):
        """Format the header to be usable by the code"""
        # 1 Retrieve the pixel scale
        pixel_key = ""
        pkey_list = ["PIXSCALE", "PIXSCALX", "SECPIX", "CDELT1", "CDELT2", "CD1_1"]
        for key in pkey_list:
            if key in self.header.keys():
                pixel_key = key
                break
        if not pixel_key:
            raise OSError("Pixel size not found in FITS file")

        pixel_scale = self.header[pixel_key]
        if pixel_key in ["CDELT1", "CDELT2", "CD1_1"]:
            pixel_scale *= 3600

        # 2 Make sure the pixel scale is only written under one key
        for pkey in pkey_list:
            self.delete_header_key(pkey)
        self.update_header_key("CD1_1", pixel_scale / 3600)
        self.update_header_key("CD1_2", 0)
        self.update_header_key("CD2_1", 0)
        self.update_header_key("CD2_2", pixel_scale / 3600)

        # 3 Update / add some keys to the header
        self.update_header_key("CTYPE1", "RA---TAN")
        self.update_header_key("CTYPE2", "DEC--TAN")
        self.update_header_key("EQUINOX", 2000.00)
        self.update_header_key("CRVAL1", 0.0)
        self.update_header_key("CRVAL2", 0.0)
        self.update_header_key("CRPIX1", self.shape[1] / 2)
        self.update_header_key("CRPIX2", self.shape[0] / 2)

        # 4 Write comment and author
        for comment_key in ["COMMENT", "comment", "Comment"]:
            self.delete_header_key(comment_key)
        self.header.add_comment("=" * 50)
        self.header.add_comment("")
        self.header.add_comment("File written using PSF tools")
        self.header.add_comment("")
        if isinstance(self.filename, str):
            basename = os.path.basename(self.filename)
            self.header.add_comment(f"Input file for computation: {basename}")
            self.header.add_comment("")
        self.header.add_comment("=" * 50)

        self.update_header_key("AUTHOR", "Alexandre Boucaud")
        self.update_header_key("INSTITUT", "IAS Orsay")

    def make_copy(self):
        """Store a copy of the input kernel and header"""
        self._image_copy = self.image.copy()
        self._header_copy = self.header.copy()

        if self.verbose:
            print_help("make_copy", "A copy of the image and header as been stored")

    def get_copy(self, header=False):
        """Return the copy of the original image

        Parameters
        ----------
        header : bool, optional
            If `True`, returns the header copy as well (default `False`)

        Returns
        -------
        tuple (`numpy.ndarray`, `fits.Header`)
            Returned only if header is `True`. A single `numpy.ndarray`
            is returned otherwise.

        """
        if header:
            return (self._image_copy, self._header_copy)
        else:
            return self._image_copy

    def restore_initial_data(self):
        """Replace current kernel and header by original version"""
        if self._image_copy is not None and self._header_copy is not None:
            self.image = self._image_copy.copy()
            self.header = self._header_copy.copy()
            if self.verbose:
                print_help(
                    "restore_initial_data", "Image and header have been restored"
                )
        else:
            print_help(
                "restore_initial_data", "No image/header backup made, cannot restore"
            )

    @property
    def shape(self):
        """Return a tuple of kernel size in x and y axes"""
        return self.image.shape

    @property
    def pixel_scale(self):
        """Retrieve the pixel scale in the image in arcseconds"""
        pixel_scale = abs(self.header["CD1_1"]) * 3600

        return round(pixel_scale, 2)

    @property
    def image_fft(self):
        """Return the Fourier transform of the current image"""
        if self._image_fft == []:
            self.compute_fft()

        return self._image_fft

    @image_fft.setter
    def image_fft(self, value):
        """Replace current fft array with new one"""
        self._image_fft = value

    @property
    def image_fft_shift(self):
        """Return the shifted Fourier transform of the image"""
        if self.image_fft == []:
            self.compute_fft()

        return np.fft.fftshift(self.image_fft)

    def compute_fft(self, real=False):
        """Perform an Fourier transform of the current image"""
        image = self.image
        image_fft = utils.fft2(image)

        if real:
            image_fft = image_fft.real

        maxval = np.abs(image_fft).max()
        if maxval > 0:
            image_fft /= maxval
        image_fft[np.abs(image_fft) < THRESH_VAL] = 0.0

        self.image_fft = image_fft

        self._circularized = False

    def compute_ifft(self, real=False):
        """Perform an inverse Fourier transform of the current image"""
        image_fft = self.image_fft
        image = utils.ifft2(image_fft)

        maxval = np.abs(image).max()
        if maxval > 0:
            image /= maxval
        image[np.abs(image) < THRESH_VAL] = 0.0

        if real:
            image = image.real

        self.image = np.abs(image)

        self._circularized = False

    def save_image(self, filename):
        """Save the current image with its header in a FITS file

        Parameters
        ----------
        filename : str
            Path where to save the image

        """
        utils.write_fits(filename, self.image, self.header)

    def save_plot(
        self,
        filepath,
        fft=False,
        log=False,
        tridim=False,
        cmap="coolwarm",
        fplot="norm",
    ):
        r"""Creates plot of the current image

        Parameters
        ----------
        filepath : str
            Output file path
        fft : bool, optional
            If `True`, plot the fft image rather than the real space one
            (default False)
        log : bool, optional
            If `True`, plot the log scaled intensity (default False)
        tridim : bool, optional
            If `True`, renders as 3d instead of 2d (default False)
        cmap : {'coolwarm', 'jet', 'gray'}, optional
            Colormap for the plot
        fplot : {'norm', 'real', 'imag'}, optional
            If `fft` is `True`, defines which complex component to plot
                * `norm`
                    Norm of the FT, \sqrt(real**2 + imag**2) (default)
                * `real`
                    Real part of the FT
                * `imag`
                    Imaginary part of the FT

        """
        shortfilepath = os.path.splitext(filepath)[0]
        n_i = 0
        while os.path.isfile(filepath):
            filepath = os.path.join(shortfilepath, f"-{n_i}.png")
            n_i += 1

        if fft:
            if fplot == "norm":
                fimage = np.abs(self.image_fft)
            elif fplot == "real":
                fimage = self.image_fft.real
            elif fplot == "imag":
                fimage = self.image_fft.imag
            else:
                raise RuntimeError("This component does not exist.")
            image = np.fft.fftshift(fimage)
        else:
            image = self.image

        image /= image.max()
        image[image < THRESH_VAL] = 0.0

        utils.plot_image(image, filepath, log=log, tridim=tridim, cm=cmap)

        if self.verbose:
            print_help("save_plot", f"Image has been saved as {filepath}")
