import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_approx_equal,
    assert_equal,
)

from psftools.classes import ImageFits

kdict = {"pixel_scale": 1.0, "backup": False, "verbose": False}
kdict_bck = {"pixel_scale": 1.0, "backup": True, "verbose": False}


def test_load_data(npdata1d, npdata2d, npdata4d):
    with pytest.raises(IOError):
        ImageFits("image.png")
        ImageFits("image.jpg")
        ImageFits("image.raw")
        ImageFits(npdata1d)
        ImageFits(npdata2d)
        ImageFits(npdata1d.tolist())
        ImageFits(npdata2d.tolist())
    with pytest.raises(ValueError):
        ImageFits(npdata1d, **kdict)
        ImageFits(npdata4d, **kdict)
        ImageFits(npdata1d.tolist(), **kdict)
        ImageFits(npdata4d.tolist(), **kdict)


def test_image(npdata2d, npdata3d):
    assert_equal(ImageFits(npdata2d, **kdict).image, npdata2d)
    assert_equal(ImageFits(npdata3d, **kdict).image, npdata3d)
    assert_equal(
        ImageFits(npdata2d.tolist(), **kdict).image, np.asarray(npdata2d.tolist())
    )
    assert_equal(
        ImageFits(npdata3d.tolist(), **kdict).image, np.asarray(npdata3d.tolist())
    )


def test_image_setter(npdata2d, npdata3d):
    imfits = ImageFits(npdata2d, **kdict)
    imfits.image = npdata3d
    #
    assert_equal(imfits.image, npdata3d)
    assert_equal(imfits.shape, npdata3d.shape)


def test_restore_initial_data(npdata2d, npdata3d):
    imfits = ImageFits(npdata2d, **kdict_bck)
    imfits.imfits = npdata3d
    imfits.restore_initial_data()
    #
    assert_equal(imfits.image, npdata2d)
    assert_equal(imfits.shape, npdata2d.shape)


def test_size(npdata2d, npdata3d):
    assert_equal(ImageFits(npdata2d, **kdict).shape, npdata2d.shape)
    assert_equal(ImageFits(npdata3d, **kdict).shape, npdata3d.shape)
    assert_equal(
        ImageFits(npdata2d.tolist(), **kdict).shape, np.asarray(npdata2d.tolist()).shape
    )
    assert_equal(
        ImageFits(npdata3d.tolist(), **kdict).shape, np.asarray(npdata3d.tolist()).shape
    )


def test_pixel_scale(npdata2d):
    assert_equal(ImageFits(npdata2d, **kdict).pixel_scale, 1.0)
