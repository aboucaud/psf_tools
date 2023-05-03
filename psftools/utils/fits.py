"""
Fits submodule
--------------
Enables I/O operations with FITS files and headers

"""
import os

from astropy.io import fits

__all__ = [
    "load_fits_with_header",
    "write_fits",
    "create_fits",
    "create_header",
    "save_fits",
    "update_header_key",
]


def load_fits_with_header(filename):
    """
    Load data and header from a FITS file

    Parameters
    ----------
    filename: str
        Name of input FITS file

    Returns
    -------
    FITS data: `numpy.ndarray`
    FITS header: `fits.header`

    """
    if not os.path.splitext(filename)[1] == ".fits":
        raise OSError("Input must be a FITS file")

    return fits.getdata(filename, header=True)


def write_fits(filename, fitsdata, fitshdr=None):
    """
    Create and save basic fits file to store output data

    Parameters
    ----------
    filename: str
        Name of output FITS file
    fitsdata: `numpy.ndarray`
        Image data
    fitshdr: `fits.header`, optional
        Header info for the image data (default `None`)

    Notes
    -----
    If the input header is set to `None`, pyfits produces
    a minimal header to accompany the data.

    """
    # if filename.endswith('.fits'):
    #     filename.rstrip('.fits')
    # filename, _ = os.path.splitext(filename)
    # filepath = os.path.join(DATAPATH, '{}.fits'.format(filename))
    # n_i = 0
    # while os.path.isfile(filepath):
    #     filepath = os.path.join(DATAPATH, '{}-{}.fits'.format(filename, n_i))
    #     n_i += 1

    nfits = fits.PrimaryHDU(data=fitsdata, header=fitshdr)
    nfits.writeto(filename)
    print(f"Output image written in {filename}")


def create_fits(image, hdr=None):
    """
    Create a basic fits file to store output data

    Parameters
    ----------
    image: `numpy.ndarray`
        Image data
    hdr: `fits.header`, optional
        Header info for the image data (default `None`)

    Returns
    -------
    hdu: `fits.hdu.image.PrimaryHDU`
        Image data embedded in a pyfits object
        with given or minimal header

    Notes
    -----
    If the input header is set to `None`, pyfits produces
    a minimal header to accompany the data.

    """
    hdu = fits.PrimaryHDU(data=image, header=hdr)

    return hdu


def create_header(image):
    """
    Create a basic header corresponding to the input data

    Parameters
    ----------
    image: `numpy.ndarray`
        Image data

    Returns
    -------
    header: `fits.Header`
        Header of the image

    """
    hdu = fits.PrimaryHDU(data=image)

    return hdu.header


def save_fits(filepath, fitsfile, copy=True):
    """
    Save a FITS file

    Parameters
    ----------
    filename: str
        Name of output FITS file
    fitsfile: `fits.hdu.image.PrimaryHDU`
        Image data embedded in a pyfits object
    copy: bool, optional
        If `True` and file already existing, add a version number
        to the filename

    """
    filename, _ = os.path.splitext(filepath)
    if os.path.isfile(filepath):
        if copy:
            n_i = 0
            while os.path.isfile(filepath):
                filepath = f"{filename}-{n_i}.fits"
                n_i += 1
        else:
            print("File already existing")
            return

    fitsfile.writeto(filepath)

    print(f"Output image written in {filepath}")


def update_header_key(header, key, value, comment="", verbose=False):
    """
    Update of add a given key of the image header

    Parameters
    ----------
    header: `fits.Header`
        Header file
    key: str
        Header key
    value: str, int, float
        The value corresponding to the header key
    comment: str, optional
        Comment on the key / value (default '')
    verbose: bool, optional
        Print information about the updated (default `False`)

    """
    if verbose:
        if key in header.keys():
            print(f"Key {key} updated")
        else:
            print(f"Key {key} added")
    if comment:
        header[key] = (value, comment)
    else:
        header[key] = value
