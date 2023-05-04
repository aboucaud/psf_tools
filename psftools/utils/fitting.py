"""
# Fitting submodule

Classes and methods for fitting distributions of PSF images
and characterizing them.

"""
import abc

import numpy as np
from scipy.optimize import leastsq
from scipy.special import j1

from psftools.utils.analysis import get_radius, integrated_profile, profile
from psftools.utils.misc import fwhm2sigma, sigma2fwhm

__all__ = ["GaussianFit", "AiryFit"]


class PSFfit:
    """Abstract class for fitting models to PSF data"""

    __metaclass__ = abc.ABCMeta

    def __init__(self, data, rotation=False):
        self.data = np.asarray(data, dtype=float)
        self.dims = self.data.ndim

        if self.dims == 1:
            self.coords = np.arange(len(self.data))
        elif self.dims == 2:
            self.coords = np.indices(self.data.shape)
        elif self.dims == 3:
            raise NotImplementedError()
        else:
            ValueError("The dimension of the data is too big")

        if self.data.size == 0:
            raise ValueError("Data is empty")
        elif self.data.size > 100000:
            self._reduce_data_size()
        else:
            self._slide = (0.0, 0.0)
            self._data_backup = np.array([[]])

        self.rotation = rotation
        self.optimized = False

    @property
    def params(self):
        """Get the current parameters of the distribution"""
        if not self.optimized:
            return self.moments

        return self._params

    @params.setter
    def params(self, param_set):
        """Set the parameters of the distribution

        Parameters
        ----------
        param_set: tuple of floats
            Parameters of the distribution

        """
        self._params = param_set

    @property
    def raw_data(self):
        if self._data_backup.size != 0:
            return self._data_backup

        return self.data

    @property
    def moments(self):
        """Compute a first guess for the distribution parameters

        Uses the first moments of the data for position and width and
        sets the angle to 0.

        """
        # Constant
        const = self.data.min()
        # Remove baseline
        bdata = self.data - const
        # Retrieve scale
        scale = bdata.max()
        # Compute moments
        total = bdata.sum()

        if self.dims == 1:
            mu_x = np.sum(self.coords * bdata) / total
            mu_y = 0.0
            sigma_x = np.sqrt(np.abs(np.sum((self.coords - mu_x) ** 2 * bdata) / total))
            sigma_y = 0.0
        else:
            x, y = self.coords
            # First moments
            mu_x = np.sum(x * bdata) / total
            mu_y = np.sum(y * bdata) / total

            col = bdata[:, mu_y]
            row = bdata[mu_x, :]

            sigma_x = np.sqrt(
                np.abs(np.sum((np.arange(col.size) - mu_y) ** 2 * col) / np.sum(col))
            )
            sigma_y = np.sqrt(
                np.abs(np.sum((np.arange(row.size) - mu_x) ** 2 * row) / np.sum(row))
            )

        return (scale, mu_x, mu_y, sigma_x, sigma_y, const, 0.0)

    @abc.abstractmethod
    def distribution(self, params):
        return np.ones_like(self.data)

    def _reduce_data_size(self, size=300):
        orig_data = self.data
        orig_shape = self.data.shape
        _, x_guess, y_guess, _, _, _, _ = self.moments
        xmin = np.max([int(x_guess) - size / 2, 0])
        xmax = np.min([xmin + size, orig_shape[0] - 1])
        if self.dims == 2:
            ymin = np.max([int(y_guess) - size / 2, 0])
            ymax = np.min([ymin + size, orig_shape[1] - 1])
            self.data = orig_data[xmin:xmax, ymin:ymax]
            self.coords = np.indices(self.data.shape)
            self._slide = (xmin, ymin)
        else:
            self.data = orig_data[xmin:xmax]
            self.coords = np.arange(self.data.size)
            self._slide[0] = xmin

        self._data_backup = orig_data.copy()

    def run_fit(self):
        def error_func(p):
            return np.ravel(self.distribution(p) - self.data)

        pfit, success = leastsq(error_func, self.params)

        if not success:
            raise ValueError("The fitting procedure did not succeed")

        self.params = pfit
        self.optimized = True

    def fitted_data(self):
        if not self.optimized:
            self.run_fit()

        return self.distribution(self.params)

    @abc.abstractmethod
    def __repr__(self):
        return "Information about the fit"

    def get_center(self):
        if not self.optimized:
            self.run_fit()

        # Add zero point if any
        add_x, add_y = self._slide

        if self.dims == 1:
            return self.params[1] + add_x
        else:
            return (self.params[1] + add_x, self.params[2] + add_y)

    @abc.abstractmethod
    def get_fwhm(self):
        if self.dims == 1:
            return -1
        else:
            return (-1, -1)

    def get_angle(self, rad=False):
        if self.dims == 1:
            raise ValueError("Cannot compute the rotation on 1D data")
        if not self.rotation:
            raise AttributeError("Rotation not allowed on __init__")

        if not self.optimized:
            self.run_fit()

        if rad:
            return np.deg2rad(self.params[-1])

        return self.params[-1]

    def get_eccentricity(self):
        if self.dims == 1:
            return 0
        else:
            b, a = self.get_fwhm()

            if a == 0.0:
                return 0

            return np.sqrt(1 - (b / a) ** 2)

    def get_ellipticity(self):
        """The ellipticity is defined here as
        |epsilon| = (a - b) / (a + b), for a > b
        """
        if self.dims == 1:
            raise ValueError("Need a two dimensional image " "to compute ellipticity ")

        if not self.rotation:
            raise AttributeError(
                "The ellipticity can only be computed " "on major and minor axes"
            )

        b, a = self.get_fwhm()

        if a + b == 0.0:
            return 0

        # return (a**2 - b**2) / (a**2 + b**2)
        return (a - b) / (a + b)

    def get_e1e2(self):
        """The real and imaginary components of the ellipticity"""
        phi_rad = self.get_angle(rad=True)
        epsilon = self.get_ellipticity()
        return epsilon * np.cos(phi_rad), epsilon * np.sin(phi_rad)

    def get_averaged_fwhm(self):
        dist, prof = profile(self.raw_data, origin=self.get_center())
        hwhm = get_radius(0.5 * prof.max(), dist, prof)

        return 2 * hwhm

    def ee2radius(self, percentage, norm_pix=-1):
        """Compute the radius corresponding to the given encercled energy (EE)

        Parameters
        ----------
        percentage: int or float
            Percentage of energy encircled in the radius
        norm_pix: int
            Index of the pixel used for the normalization of the
            integrated profile. (default -1 = last pixel)

        Returns
        -------
        radius: float
            Radius encircling the given energy

        """
        if (percentage <= 0) or (percentage > 1):
            raise ValueError("Input energy percentage should be " "between 0 and 1")
        dist, iprof = integrated_profile(self.raw_data, origin=self.get_center())
        iprof /= iprof[norm_pix]

        return get_radius(percentage, dist, iprof)

    def radius2ee(self, hwhm_frac, norm_pix):
        """Compute the encercled energy corresponding to the given radius

        The radius here is defined as a fraction of the HWHW of the PSF.

        Parameters
        ----------
        hwhm_frac: int or float
            Fraction of the HWHM
        norm_pix: int
            Index of the pixel used for the normalization of the
            integrated profile. (default -1 = last pixel)

        Returns
        -------
        ee: float
            Encircled energy

        """
        if hwhm_frac < 0:
            raise ValueError("The hwhm fraction should be positive")

        hwhm = self.get_averaged_fwhm() / 2
        radius = hwhm_frac * hwhm

        dist, iprof = integrated_profile(self.raw_data, origin=self.get_center())
        iprof /= iprof[norm_pix]

        ee = get_radius(radius, iprof, dist)

        return ee


# Sub-classes
# -----------


class GaussianFit(PSFfit):
    """Sub class of PSFfit dealing with Gaussian distributions"""

    def distribution(self, params):
        # Unwrap parameters
        scale, mu_x, mu_y, sigma_x, sigma_y, const, theta = params

        if self.dims == 1:
            return scale * np.exp(-0.5 * ((self.coords - mu_x) / sigma_x) ** 2) + const
        else:
            x, y = self.coords
            if self.rotation:
                # Convert angle to radians
                rtheta = np.deg2rad(theta)
                xrot = (x - mu_x) * np.cos(rtheta) - (y - mu_y) * np.sin(rtheta)
                yrot = (x - mu_x) * np.sin(rtheta) + (y - mu_y) * np.cos(rtheta)
                return (
                    scale
                    * np.exp(-0.5 * ((xrot / sigma_x) ** 2 + (yrot / sigma_y) ** 2))
                    + const
                )
            else:
                return (
                    scale
                    * np.exp(
                        -0.5
                        * (((x - mu_x) / sigma_x) ** 2 + ((y - mu_y) / sigma_y) ** 2)
                    )
                    + const
                )

    def get_fwhm(self):
        if not self.optimized:
            self.run_fit()

        if self.dims == 1:
            return sigma2fwhm(self.params[3])
        else:
            return np.sort(map(sigma2fwhm, self.params[3:5]))

    def __repr__(self):
        if not self.optimized:
            self.run_fit()

        scale, mu_x, mu_y, sigma_x, sigma_y, const, theta = self.params

        text = [
            "Gaussian fit results",
            "--------------------",
            f"type of image: {self.dims}D array",
        ]

        if self.dims == 1:
            text += [
                f"center position (pixel index) = {self.get_center():.1f}",
                f"FWHM = {sigma2fwhm(sigma_x):.2f} pixels (sigma = {sigma_x:.2f})",
            ]
        else:
            ix, iy = self.get_center()
            text += [
                f"rotation allowed: {self.rotation}",
                f"center position (pixel indices) = ({ix:1.1f}, {iy:1.1f})",
                f"FWHM x = {sigma2fwhm(sigma_x):.2f} pixels (sigma_x = {sigma_x:.2f})",
                f"FWHM y = {sigma2fwhm(sigma_y):.2f} pixels (sigma_y = {sigma_y:.2f})",
            ]

            if self.rotation:
                # Due to symmetries, the rotation angle should be < 180 deg
                norm_angle = np.mod(theta, 180)
                text.append(f"theta = {norm_angle:.2f} deg")

        return "\n".join(text)


class AiryFit(PSFfit):
    """Sub class of PSFfit dealing with Airy distributions"""

    def distribution(self, params):
        # Unwrap parameters
        scale, mu_x, mu_y, fwhm_x, fwhm_y, const, theta = params

        if self.dims == 1:
            dist = np.abs(self.coords - mu_x)
            # dist[dist==0] = 1.e-30
            dist *= 1.61633 / (fwhm_x / 2)
        else:
            x, y = self.coords
            if self.rotation:
                # Convert angle to radians
                rtheta = np.deg2rad(theta)
                coord_x = (x - mu_x) * np.cos(rtheta) - (y - mu_y) * np.sin(rtheta)
                coord_y = (x - mu_x) * np.sin(rtheta) + (y - mu_y) * np.cos(rtheta)
            else:
                coord_x = np.abs(x - mu_x)
                coord_y = np.abs(y - mu_y)

            # coord_x[coord_x==0] = 1.e-30
            coord_x *= 1.61633 / (fwhm_x / 2)
            # coord_y[coord_y==0] = 1.e-30
            coord_y *= 1.61633 / (fwhm_y / 2)

            dist = np.hypot(coord_x, coord_y)

        airy = (2 * j1(dist) / dist) ** 2
        airy[dist == 0] = 1.0

        return scale * airy + const

    def get_fwhm(self):
        if not self.optimized:
            self.run_fit()

        if self.dims == 1:
            return self.params[3]
        else:
            return np.sort(self.params[3:5])

    def get_diff_firstlobe(self):
        if self.dims != 1:
            raise NotImplementedError("The method only works for 1d data")

        if not self.optimized:
            self.run_fit()

        coords = self.coords - self.get_center()

        min1 = 1.185 * self.get_fwhm()
        min2 = 2.17 * self.get_fwhm()
        cond_pos = (coords > min1) & (coords < min2)
        cond_neg = (coords < -min1) & (coords > -min2)

        fit = self.fitted_data()
        data = self.data

        neg_diff = np.sum(data[cond_neg] - fit[cond_neg])
        pos_diff = np.sum(data[cond_pos] - fit[cond_pos])

        return neg_diff, pos_diff

    def __repr__(self):
        if not self.optimized:
            self.run_fit()

        scale, mu_x, mu_y, fwhm_x, fwhm_y, const, theta = self.params

        text = [
            "Airy fit results",
            "----------------",
            f"type of image: {self.dims}D array",
        ]

        if self.dims == 1:
            text += [
                f"center position (pixel index) = {self.get_center():.1f}",
                f"FWHM = {fwhm_x:.2f} pixels (sigma = {fwhm2sigma(fwhm_x):.2f})",
            ]
        else:
            ix, iy = self.get_center()
            text += [
                f"rotation allowed: {self.rotation}",
                f"center position (pixel indices) = ({ix:1.1f}, {iy:1.1f})",
                f"FWHM x = {fwhm_x:.2f} pixels (sigma_x = {fwhm2sigma(fwhm_x):.2f})",
                f"FWHM y = {fwhm_y:.2f} pixels (sigma_y = {fwhm2sigma(fwhm_y):.2f})",
            ]

            if self.rotation:
                # Due to symmetries, the rotation angle should be < 180 deg
                norm_angle = np.mod(theta, 180)
                text.append(f"theta = {norm_angle:.2f} deg")

        return "\n".join(text)
