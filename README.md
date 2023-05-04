
PSF tools
=========

[![Tests][gh-workflow-badge]][gh-workflow]
[![License][license-badge]](LICENSE)
[![pre-commit][precommit-badge]][precommit]
<!-- ![Python supported versions][pyversion-badge] -->
<!-- [![PyPI][pypi-badge]][pypi] -->

[gh-workflow]: https://github.com/aboucaud/psf_tools/actions/workflows/python-package.yml
[gh-workflow-badge]: https://github.com/aboucaud/psf_tools/actions/workflows/python-package.yml/badge.svg
[license-badge]: https://img.shields.io/github/license/aboucaud/psf_tools?color=blue
[precommit]: https://github.com/aboucaud/psf_tools/actions/workflows/check_style.yml
[precommit-badge]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
[pyversion-badge]: https://img.shields.io/pypi/pyversions/psf_tools?color=yellow&logo=pypi
[pypi-badge]: https://badge.fury.io/py/psf_tools.svg
[pypi]: https://pypi.org/project/psf_tools/

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

API docs
--------
The basic API docs are available [here][docs]

[docs]: https://aboucaud.github.io/psf_tools

Acknowledgement
---------------
Some of this work has benefited from daily exchanges with Hervé Dole (IAS), Alain Abergel (IAS) and François Orieux (L2S, Centrale Supélec).

License
-------
The code is released under the [MIT license](LICENSE).
