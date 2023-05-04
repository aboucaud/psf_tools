"""
# PSF submodule

Class handling PSF FITS image and header and various operations on them.

"""
import numpy as np
from scipy.ndimage import zoom

import psftools.utils as utils
from psftools.classes.core import ImageFits
from psftools.utils import circ_psd, print_help, profile

__all__ = ["PSF"]

THRESH_VAL = 1.0e-8


class PSF(ImageFits):
    """
    Subclass of ImageFits for handling PSF images

    Parameters
    ----------
    filename : str
        Path to the image FITS file
    pixel_scale : float, optional (default 0.0)
        Pixel scale of input image data in arcseconds
    backup : bool
        Create a backup when loading the data
    verbose : bool, optional
        Add some verbosity to the actions

    Attributes
    -----------------
    filename: str
        Path to the image FITS file
    image: array_like
        Image data cube
    header: `fits.Header`
        Image header
    image_fft: array_like
        Array of the Fourier transform of the images
    fwhm: float
        The FWHM value of the PSF
    verbose: bool
        If `True`, print debug messages
    _image_copy: array_like, optional
        Backup image if backup=`True`
    _header_copy: `fits.Header`, optional
        Backup header if backup=`True`
    _circularized: bool
        If `True`, indicates the image has already been circularized

    """

    def __init__(self, filename, pixel_scale=0.0, backup=True, verbose=False):
        ImageFits.__init__(self, filename, pixel_scale, backup, verbose)
        self.image = np.nan_to_num(self.image)

        self.fwhm = 0.0
        self._circularized = False

        self.update_header_key("UNITS", "counts")

    @property
    def is_square(self):
        """Return `True` if the image array is a square"""
        return self.shape[0] == self.shape[1]

    @property
    def is_circularized(self):
        """Return `True` if the image has not been modified
        since last circularization"""
        return self._circularized

    @property
    def is_centered(self):
        """Return `True` if central pixel corresponds to the PSF peak"""
        centerpix = self.shape[0] // 2
        return utils.find_peak_position(self.image) == (centerpix, centerpix)

    @property
    def center_distance(self):
        """Return the euclidian pixel distance to the array center"""
        return utils.center_distance(self.shape[0])

    @property
    def fwhm(self):
        """Return the FWHM (Full Width at Half Maximum) of the PSF"""
        if self._fwhm == 0.0:
            self._compute_fwhm()
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        """FWHM setter function

        Parameters
        ----------
        value: int
            Value of the new FWHM

        """
        self._fwhm = value

    def _compute_fwhm(self, circularize=False):
        """Compute the fwhm of the psf from the circularized image"""
        # Make axi-symmetric before computing the fwhm
        if circularize:
            if not self._circularized:
                self.circularize()
        # Extract psf profile
        # resdict = utils.analyze_radial_profile(self.image)
        radius, profil = profile(self.image)
        # Interpolate to find the half maximum value
        # rad = resdict['radius'] * self.pixel_scale
        # profil = resdict['profile']
        radius *= self.pixel_scale
        hwhm = utils.get_radius(0.5 * np.max(profil), radius, profil)

        self.fwhm = round(2 * hwhm, 3)

    def normalize(self, to_max=False):
        """Kernel normalization

        Parameters
        ----------
        to_max: bool, optional
            If true, normalize to the maximum value in the array
            (default False)

        """
        if to_max:
            imax = self.image.max()
            if imax != 0:
                self.image /= imax
                msg = "Maximum pixel value normalized to unity"
            else:
                raise ValueError("Maximum pixel value is zero")
        else:
            isum = self.image.sum()
            if isum != 0:
                self.image /= isum
                msg = "Pixel sum normalized to unity"
            else:
                raise ValueError("Sum of pixel values is zero")

        if self.verbose:
            print_help("normalize", msg)

    def clean_circle(self):
        """Clean image outside of inner circle"""
        # if (self.shape[0] % 2 == 0) or (self.shape[1] % 2 == 0):
        self.make_odd_square()
        # self.center_psf()

        wh_circle = self.center_distance < self.shape[0] // 2
        cleaned_image = np.zeros_like(self.image)
        cleaned_image[wh_circle] = self.image[wh_circle]

        self.image = cleaned_image

    def clean_threshold(self, thresh=THRESH_VAL):
        """Clean image under a certain threshold"""
        if (self.shape[0] % 2 == 0) or (self.shape[1] % 2 == 0):
            self.make_odd_square()
            self.center_psf()

        self.normalize(to_max=True)

        wh_thresh = self.image > thresh
        cleaned_image = np.zeros_like(self.image)
        cleaned_image[wh_thresh] = self.image[wh_thresh]

        self.image = cleaned_image

    def make_odd_square(self, size=0):
        """Pad kernel to get a square image of odd side size
        A given size can be optionnally input

        Parameters
        ----------
        size: int, optional
            A size to pad the image

        """
        x_size, y_size = self.shape
        # define square size
        square_size = max(x_size, y_size)
        if size:
            square_size = max(square_size, size)
        # ensure odd
        if square_size % 2 == 0:
            square_size += 1

        if square_size > x_size or square_size > y_size:
            self.resize(square_size)

    def resize(self, new_size):
        """
        Resize the PSF without resampling

        Warning: Can lead to a loss of information if used unproperly !
                 Make sure the new size has a meaning

        Parameters
        ----------
        new_size: int, odd number
            Output size of the image

        """
        new_size = int(new_size)
        if new_size % 2 == 0:
            new_size += 1
        if (new_size != self.shape[0]) or (new_size != self.shape[1]):
            # Store the initial flux value
            flux_old = self.image.sum()

            old_size = self.shape
            if new_size < min(old_size):
                new_image = utils.trim(self.image, (new_size, new_size))
                msg = "Image trimmed from {0}x{1} to {2}x{2} pixels."
            elif new_size > max(old_size):
                new_image = utils.zero_pad(
                    self.image, (new_size, new_size), position="center"
                )
                msg = "Image padded from {0}x{1} to {2}x{2} pixels."
            else:
                new_image = utils.zero_pad(
                    self.image, (max(old_size), max(old_size)), position="center"
                )
                msg = "Image padded from {0}x{1} to {2}x{2} pixels."

            new_image[np.abs(new_image) < 1.0e-10 * abs(new_image.max())] = 0.0

            self.image = new_image
            self.update_header_key("NAXIS1", new_size)
            self.update_header_key("NAXIS2", new_size)
            self.update_header_key("CRPIX1", new_size // 2)
            self.update_header_key("CRPIX2", new_size // 2)

            self.normalize(to_max=True)

            flux_new = self.image.sum()
            flux_loss = 100.0 * (flux_new - flux_old) / flux_old
            if abs(flux_loss) < 1.0e-4:
                flux_loss = 0.0
            flux_msg = "The total flux in the image changed by {}%"

            if self.verbose:
                print_help("resize", msg.format(old_size[0], old_size[1], new_size))
                print_help("resize", flux_msg.format(flux_loss))

    def resample(self, new_pixel_scale, interp_order=2):
        """
        Resample kernel to a given pixel scale

        Parameters
        ----------
        new_pixel_scale : float
            Pixel scale in arcseconds to which resample the kernel
        interp_order : int, optional
            Spline interpolation order [0, 5] (default 2: cubic)

        """
        if not self.is_square:
            self.make_odd_square()

        if round(new_pixel_scale, 2) != self.pixel_scale:
            old_pixel_scale = self.pixel_scale
            old_size = self.shape[0]

            new_size_raw = old_size * old_pixel_scale / new_pixel_scale
            new_size = int(round(new_size_raw))

            # New size must be odd
            if new_size % 2 == 0:
                if new_size_raw < new_size:
                    new_size += 1
                else:
                    new_size -= 1

            self.image = zoom(self.image, new_size / old_size, order=interp_order)

            self.update_header_key("CD1_1", new_pixel_scale / 3600)
            self.update_header_key("CD1_2", 0)
            self.update_header_key("CD2_1", 0)
            self.update_header_key("CD2_2", new_pixel_scale / 3600)
            self.update_header_key("NAXIS1", new_size)
            self.update_header_key("NAXIS2", new_size)
            self.update_header_key("CRPIX1", new_size // 2)
            self.update_header_key("CRPIX2", new_size // 2)

            msg = "PSF has been resampled from {} arcsec/pix to {} arcsec/pix"
            if self.verbose:
                print_help("resample", msg.format(old_pixel_scale, self.pixel_scale))

    def center_psf(self):
        """Shift the image to put the maximum at the center"""
        self.image = utils.center_psf(self.image)
        # self._circularized = False

        if self.verbose:
            print_help("center_psf", "PSF peak centered")

    def rotate(self, angle):
        """Rotate image from North to East given the angle in degrees

        Parameters
        ----------
        angle: float
            Angle in degrees

        """
        if (not self.is_centered) or (not self.is_square):
            self.center_psf()

        self.image = utils.imrotate(self.image, angle)

    def plot_profile(self, outpng=""):
        """Plot the radial profile of the PSF"""
        import matplotlib.pyplot as plt

        # d = utils.analyze_radial_profile(self.image)
        rad, prof = profile(self.image)
        fig, axis = plt.subplots(figsize=(12, 9))
        # axis.plot(d['radius'] * self.pixel_scale, np.log10(d['profile']))
        axis.plot(rad * self.pixel_scale, np.log10(prof))
        axis.set_xlabel("radius [arcsec]")
        axis.set_ylabel(r"profile [$\log_{10}$]")
        if outpng:
            fig.savefig(outpng)

    def plot_fft_profile(self, real=False, outpng=""):
        """Plot the radial profile of the Fourier transform of the PSF"""
        import matplotlib.pyplot as plt

        image = self.image_fft_shift
        if real:
            image = image.real
        # d = utils.analyze_radial_profile(np.abs(image))
        rad, prof2 = circ_psd(self.image)
        fig, axis = plt.subplots(figsize=(12, 9))
        # axis.plot(d['radius'] * self.pixel_scale, np.log10(d['profile']))
        axis.plot(rad, np.log10(np.sqrt(prof2)))
        axis.set_xlabel("k")
        axis.set_ylabel(r"profile [$\log_{10}$]")
        if outpng:
            fig.savefig(outpng)

    def plot_image(self, outpng="", colorbar=True, axes=True, trans=False):
        """Show the image in decimal log space"""
        import matplotlib.pyplot as plt

        axis = plt.subplot()
        img = axis.imshow(np.log10(self.image), vmin=-6, vmax=0)
        if not axes:
            plt.axis("off")
        if colorbar:
            plt.colorbar(img)
        if outpng:
            plt.savefig(outpng, transparent=trans)

    def plot_fft(self, outpng=""):
        """Show the Fourier transform of the image in decimal log space"""
        import matplotlib.pyplot as plt

        axis = plt.subplot()
        img = axis.imshow(np.log10(np.abs(self.image_fft_shift)))
        plt.colorbar(img)
        if outpng:
            plt.savefig(outpng)

    def circularize(self, log=False):
        """Apply a circularization procedure to the image

        Parameters
        ----------
        log: bool
            If `True`, compute the circularization in log space
            (default False)

        """
        if not self.is_square:
            self.make_odd_square()

        if not self.is_centered:
            self.center_psf()

        self.image = self.circularize_psf_aniano(log=log)
        # self.image = self.circularize_psf_chanial()

        # make sure it is centered
        if not self.is_centered:
            print_help("circularize", "The circularization off-centered the image")

        self._circularized = True

        if self.verbose:
            print_help("circularize", "PSF circularized")

    def circularize_psf_aniano(
        self, n_itermax=14, interp_order=2, log=True, correction=False
    ):
        """ "
        Aniano circularization procedure

        We successively rotate the image and add it to itself
        with angles decreasing by a factor 2 every iteration.
        This is numerically equivalent to computing the average
        over 2^n rotations where n is the number of iterations.
        The default choice of n=14 hence produces a PSF invariant
        under rotations of any angle that is a multiple of 360/2^14
        = 360/16384 ~ 0.022 deg

        Parameters
        ----------
        n_itermax : int, optional
            Number of performed rotational iterations (default 14)
        interp_order : int, optional
            Order of spline interpolation (default 2 = cubic)
        log: bool
            If `True`, compute the circularization in log space
            (default True)
        correction: bool
            If `True`, measures the distortion induced by
            the circularization and apply it back to the array
            (default `False`)

        """
        # actual_size = self.shape[0]
        # int((float(actual_size) * np.sqrt(2.0) - actual_size) / 2.0) + 1
        psf = self.image
        psf_circle = utils.circularize(psf, n_itermax, interp_order, log)

        if correction:
            psf_ones = np.ones(self.shape)
            psf_ones_circle = utils.circularize(
                psf_ones, n_itermax, interp_order, log=False
            )
            valid = psf_ones_circle > 0.1
            if np.any(valid):
                circularization_filter = np.zeros(self.shape)
                circularization_filter[valid] = 1 / psf_ones_circle[valid]
                psf_circle *= circularization_filter

        # The pixels outside the major circle contained in the image
        # have to be masked out since they do not contain useful info
        distance = self.center_distance
        psf_circle[distance > self.shape[0] // 2] = 0.0

        psf_circle[np.abs(psf_circle) < THRESH_VAL] = 0.0

        return psf_circle

    def circularize_psf_chanial(self):
        """Circularize image using the 2D projection of its profile"""
        radius, profil = profile(self.image)
        distance = self.center_distance
        psf_circle = np.interp(distance, radius, profil)

        # CUTS
        psf_circle[distance > self.shape[0] // 2] = 0.0
        psf_circle[np.abs(psf_circle) < THRESH_VAL] = 0.0

        return psf_circle

    def circularize_fft(self):
        """Circularize the Fourier transform of the PSF"""
        fft = self.image_fft
        fft_phase = np.arctan2(fft.imag, fft.real)

        fft_amp = self.circularize_fft_aniano()
        # fft_amp = self.circularize_fft_chanial()

        self.image_fft = fft_amp * np.exp(1j * fft_phase)

        if self.verbose:
            print_help("circularize", "PSF Fourier circularized")

    def circularize_fft_aniano(self, n_itermax=14, interp_order=2):
        """Circularize the Fourier transform of the PSF"""
        amp_fft_shift = np.abs(self.image_fft_shift)
        # circularization
        amp_fft_shift_circle = utils.circularize(amp_fft_shift, n_itermax, interp_order)
        # cleaning
        distance = self.center_distance
        amp_fft_shift_circle[distance > self.shape[0] // 2] = 0.0

        return np.fft.ifftshift(amp_fft_shift_circle)

    def circularize_fft_chanial(self):
        """Circularize FT image using the 2D projection of its profile"""
        radius, profil = circ_psd(self.image)
        distance = self.center_distance
        amp_fft_shift_circle = np.interp(
            distance / self.shape[0], radius, np.sqrt(profil)
        )
        amp_fft_shift_circle[distance > self.shape[0] // 2] = 0.0

        return np.fft.ifftshift(amp_fft_shift_circle)

    def filter_fft(self):
        """
        Low-pass filter to remove high frequency components in the Fourier
        transform of the kernels

        Notes
        -----
        The high frequency cut k_hf = 4 x 2pi / FWHM has been found
        empirically by Aniano et al. 2011

        """
        amp_fft_shift = np.abs(self.image_fft_shift)
        fft_phase = np.arctan2(self.image_fft.imag, self.image_fft.real)
        # Fourier modes (omitting the 2pi term)
        k_array = self.center_distance * self.pixel_scale / self.shape[0]
        k_hi = 4 / self.fwhm
        k_lo = 0.9 * k_hi

        filter_mask = np.exp(-1.0 * (1.8249 * (k_array - k_lo) / (k_hi - k_lo)) ** 4)
        filter_mask[k_array < k_lo] = 1.0
        filter_mask[k_array >= k_hi] = 0.0

        amp_fft_shift_filt = amp_fft_shift * filter_mask

        # cleaning
        amp_fft_shift_filt[np.abs(amp_fft_shift_filt) < THRESH_VAL] = 0.0

        amp_fft_filtered = np.fft.ifftshift(amp_fft_shift_filt)
        self.image_fft = amp_fft_filtered * np.exp(1j * fft_phase)

        if self.verbose:
            msg = "PSF Fourier transform filtered above k_cutoff = {}"
            print_help("filter_fft", msg.format(2 * np.pi * k_hi))

    def get_inverse_fft_image(self):
        """
        The inverse of the Fourier image.

        In order to avoid infinites in the output array,
        we set to zero all the output values that correspond
        to input values whose norm is less that 1.e-7 times
        the peak value.

        Returns
        -------
        fft_inverse: complex `numpy.ndarray`
            The inverse of the current Fourier image

        Notes
        -----
        The high frequency cut-off for the smooth filter is determined
        as the frequency above which FT[PSF](k) < 5.0e-3 max(FT[PSF])

        """
        amp_fft = np.abs(self.image_fft)
        wh_good = amp_fft / amp_fft.max() > 1.0e-7
        fft_inverse = np.zeros_like(self.image_fft)
        fft_inverse[wh_good] = 1 / self.image_fft[wh_good]
        # phase back up
        phase = np.arctan2(fft_inverse.imag, fft_inverse.real)
        amp_fft_inverse_shifted = np.fft.fftshift(np.abs(fft_inverse))

        # cutoff frequency determination
        # fft_dict = utils.analyze_radial_profile(np.fft.fftshift(amp_fft))
        # fft_radius = fft_dict['radius']
        # fft_profile = fft_dict['profile']
        fft_radius, fft_profile = profile(np.fft.fftshift(amp_fft))

        # Fourier modes (omitting the 2pi term)
        # k_array = self.center_distance * self.pixel_scale / self.shape[0]
        k_array = self.center_distance / self.shape[0]
        k_hi = utils.get_radius(5.0e-3 * fft_profile.max(), fft_radius, fft_profile)
        k_hi /= self.shape[0]
        k_lo = 0.7 * k_hi
        # filter creation
        filter_mask = 0.5 * (1 + np.cos(np.pi * (k_array - k_lo) / (k_hi - k_lo)))
        filter_mask[k_array < k_lo] = 1.0
        filter_mask[k_array >= k_hi] = 0.0
        # filtering
        amp_fft_shift_filt = amp_fft_inverse_shifted * filter_mask
        # cleaning
        amp_fft_shift_filt[np.abs(amp_fft_shift_filt) < THRESH_VAL] = 0.0

        amp_fft_filt = np.fft.ifftshift(amp_fft_shift_filt)
        fft_inverse_filt = amp_fft_filt * np.exp(1j * phase)

        return fft_inverse_filt

    def correct_wings(self, r_min, r_max, r_study, n_itermax=10):
        """
        Fit a model to the wings of the PSF and add it to the core image

        Parameters
        ----------
        r_min: float
            Lower bound in radius
        r_max: float
            Upper bound in radius
        r_study: float
            Transition radius
        n_itermax: int, optional
            Number of iterations for the fitting process

        """
        # Check and correct for unappropriate values
        r_min = min(max(r_min, 1), 1000)
        r_max = min(max(r_max, 1), 1000)
        r_max = max(r_min + 1, r_max)
        r_study = min(max(r_study, r_min), r_max)

        # Cuts to avoid log issues
        d_min = 1.0
        v_min = THRESH_VAL

        # Retrieve radius and profile
        anadict = utils.analyze_radial_profile(self.image)
        distance = anadict["radius"] * self.pixel_scale
        distance[distance <= d_min] = d_min
        wh_analysis = (distance > r_min) & (distance < r_max)

        value = anadict["profile"]
        value[value <= v_min] = v_min

        ldist = np.log10(distance)
        lval = np.log10(value)
        wh_int = wh_analysis.astype(np.int)

        # PSF wing profile : r^-3 + C
        index = -3.0

        def wing_model(rad, index, const):
            return rad**index * 10**const

        n_iter = 0
        keep_going = True
        while keep_going and (n_iter < n_itermax):
            n_points = np.sum(wh_int)
            const = (np.sum(3.0 * ldist * wh_int) + np.sum(lval * wh_int)) / n_points
            # msg = """Using {} points, PSF = {} * r^-3"""
            # print(msg.format(n_points, const))

            # Data rejection
            difference = (lval - (index * ldist + const)) * wh_int
            std = np.sqrt(np.sum(difference**2) / n_points)
            wh_reject = np.abs(difference) > 1.8 * std

            if np.any(wh_reject):
                wh_int[wh_reject] = 0
            else:
                keep_going = False
            n_iter += 1

        # print("Wing model:")
        # print('10^{} * r^-3'.format(const))

        # After 1D analysis, 2D realization
        # grid_size = max(output_size, self.shape[0])
        grid_size = self.shape[0]
        gcenter = grid_size // 2
        grid_dist = utils.center_distance(grid_size) * self.pixel_scale
        grid_dist[gcenter, gcenter] = self.pixel_scale
        modeled_wings = wing_model(grid_dist, index, const)

        if grid_size > self.shape[0]:
            # reshape psf
            self.resize(grid_size)

        new_image = utils.merge_images(
            self.image, modeled_wings, r_study / self.pixel_scale
        )

        self.image = new_image

        if self.verbose:
            print_help("correct_wings", "Wings fit and added to the core PSF shape")
