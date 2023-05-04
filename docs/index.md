# Welcome to the PSF tools API documentation

`psftools` is a personal library of utility methods to deal with astronomical point spread functions (PSFs).

In particular one finds methods to analyse the characteristics of a PSF such as its encircled energy, azimuthal profiles, ellipticity, etc., and perform classical operations on them such a normalisation, conversion to Fourier domain, convolutions, etc.

The library installs two convenient scripts:
    - `make_kernel` to compute a homogenization kernel from one PSF image to a broader one (not handling actual deconvolution)
    - `homogenization` to apply these kernels on astronomical images to perform PSF matching.

It has been mainly developed at Institut d'Astrophysique Spatiale (2014-2016) and was rejuvenated in May 2023 to deal with current Python standards and prepare its release on GitHub.
