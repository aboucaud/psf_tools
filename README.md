
PSF tools
=========

`psftools` is a personal library of utility methods to deal with astronomical point spread functions (PSFs).

In particular one finds methods to analyse the characteristics of a PSF such as its encircled energy, azimuthal profiles, ellipticity, etc., and perform classical operations on them such a normalisation, conversion to Fourier domain, convolutions, etc.

The library installs two convenient scripts:
    - `make_kernel` to compute a homogenization kernel from one PSF image to a broader one (not handling actual deconvolution)
    - `homogenization` to apply these kernels on astronomical images to perform PSF matching.

It has been mainly developed during my engineer position at Institut d'Astrophysique Spatiale (2014-2016) and was rejuvenated in May 2023 to deal with current Python standards and prepare its release on GitHub.

Getting started
---------------
```
pip install git+https://github.com/aboucaud/psf_tools.git
```

Acknowledgement
---------------
Some of this work has benefited from daily exchanges with Hervé Dole (IAS), Alain Abergel (IAS) and François Orieux (L2S, Centrale Supélec).

License
-------
The code is licensed under the [MIT license](LICENCE).
