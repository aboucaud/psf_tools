import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_approx_equal,
    assert_equal,
)

from psftools.classes import PSF

npempty = np.array([[], []])
npdata1d = np.arange(10)
npdata2d = np.arange(25).reshape(5, 5)
npdata3d = np.arange(64).reshape(4, 4, 4)
npdata4d = np.arange(81).reshape(3, 3, 3, 3)
listempty = [[], []]
listdata1d = [1, 2, 3, 4, 5, 6, 7, 8, 9]
listdata2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
listdata3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
listdata4d = [
    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
    [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
]

kdict = {"pixel_scale": 1.0, "backup": False, "verbose": False}
kdict_bck = {"pixel_scale": 1.0, "backup": True, "verbose": False}


def test_load_data():
    with pytest.raises(IOError):
        PSF("image.png")
        PSF("image.jpg")
        PSF("image.raw")
        PSF(npdata1d)
        PSF(npdata2d)
        PSF(listdata1d)
        PSF(listdata2d)
    with pytest.raises(ValueError):
        PSF(npempty, **kdict)
        PSF(npdata1d, **kdict)
        PSF(npdata4d, **kdict)
        PSF(listempty, **kdict)
        PSF(listdata1d, **kdict)
        PSF(listdata4d, **kdict)


def test_image():
    assert_equal(PSF(npdata2d, **kdict).image, npdata2d)
    assert_equal(PSF(npdata3d, **kdict).image, npdata3d)
    assert_equal(PSF(listdata2d, **kdict).image, np.asarray(listdata2d))
    assert_equal(PSF(listdata3d, **kdict).image, np.asarray(listdata3d))


def test_image_setter():
    imfits = PSF(npdata2d, **kdict)
    imfits.image = npdata3d
    #
    assert_equal(imfits.image, npdata3d)
    assert_equal(imfits.shape, npdata3d.shape)


def test_restore_initial_data():
    imfits = PSF(npdata2d, **kdict_bck)
    imfits.imfits = npdata3d
    imfits.restore_initial_data()
    #
    assert_equal(imfits.image, npdata2d)
    assert_equal(imfits.shape, npdata2d.shape)


def test_size():
    assert_equal(PSF(npdata2d, **kdict).shape, npdata2d.shape)
    assert_equal(PSF(npdata3d, **kdict).shape, npdata3d.shape)
    assert_equal(PSF(listdata2d, **kdict).shape, np.asarray(listdata2d).shape)
    assert_equal(PSF(listdata3d, **kdict).shape, np.asarray(listdata3d).shape)


def test_pixel_scale():
    assert_equal(PSF(npdata2d, **kdict).pixel_scale, 1.0)
