import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_approx_equal,
    assert_equal,
)

from psftools.utils.image import *

errshape = "incorrect shape"
errout = "incorrect output"
errval = "incorrect value"


def test_convolve():
    pass


def test_trim(tones):
    arr, size = tones
    size_e = size - 1
    size_o = size - 2
    shape_ee = (size_e, size_e)
    shape_eo = (size_e, size_o)
    shape_oo = (size_o, size_o)

    assert_equal(trim(arr, (size, size)), arr, errout)
    # Shapes
    assert_equal(trim(arr, shape_oo).shape, shape_oo, errshape)
    assert_equal(trim(arr, np.asarray(shape_oo, float)).shape, shape_oo, errshape)
    # Wrong inputs
    with pytest.raises(ValueError):
        trim(arr, 0)
        trim(arr, (size_e, 0))
        trim(arr, (size + 1, size))
        trim(arr, shape_ee)
        trim(arr, shape_eo)


def test_zero_pad(tones):
    arr, size = tones
    size_e = size + 1
    size_o = size + 2
    shape_ee = (size_e, size_e)
    shape_eo = (size_e, size_o)
    shape_oo = (size_o, size_o)

    assert_equal(zero_pad(arr, (size, size)), arr, errout)
    # Shapes
    for pos in ["corner", "center"]:
        assert_equal(zero_pad(arr, shape_ee, "corner").shape, shape_ee, errshape)
        assert_equal(zero_pad(arr, shape_oo, pos).shape, shape_oo, errshape)
        assert_equal(zero_pad(arr, shape_eo, "corner").shape, shape_eo, errshape)
        # Wrong inputs
        with pytest.raises(ValueError):
            zero_pad(arr, -2, position=pos)
            zero_pad(arr, size_e, position=pos)
            zero_pad(arr, (size - 1, size_e), pos)
            zero_pad(arr, shape_ee, "center")
            zero_pad(arr, shape_eo, "center")


def test_mirror_pad(tones):
    arr, size = tones
    even = 4
    odd = 5
    size_even = size + 2 * even
    size_odd = size + 2 * odd
    assert_equal(mirror_pad(arr, 0), arr, errout)
    # Shapes
    assert_equal(mirror_pad(arr, even).shape, (size_even, size_even), errshape)
    assert_equal(mirror_pad(arr, float(even)).shape, (size_even, size_even), errshape)
    assert_equal(mirror_pad(arr, even, even).shape, (size_even, size_even), errshape)
    assert_equal(mirror_pad(arr, odd).shape, (size_odd, size_odd), errshape)
    assert_equal(mirror_pad(arr, odd, odd).shape, (size_odd, size_odd), errshape)
    assert_equal(mirror_pad(arr, even, odd).shape, (size_even, size_odd), errshape)
    # Wrong inputs
    with pytest.raises(ValueError):
        mirror_pad(arr, -2)
        mirror_pad(arr, size + 1)
        mirror_pad(arr, size - 1, size + 1)


# def test_psf_peak_position(tgrid):
#     x, y, cpix = tgrid
#     delta_x = 2
#     delta_y = -3
#     # Simple case
#     img_xy = np.exp(-0.01*((x-cpix+delta_x)**2 +
#                            (y-cpix+delta_y)**2))
#     assert_equal(find_peak_position(img_xy), (cpix-delta_x, cpix-delta_y))
#     # Saturated case
#     img_saturated1 = img_xy.copy()
#     img_saturated1[img_xy > 0.97] = 1.0
#     assert_equal(find_peak_position(img_saturated1),
#                      (cpix-delta_x, cpix-delta_y))


def test_center_psf(tgrid):
    x, y, cpix = tgrid
    img = np.exp(-0.01 * ((x - cpix) ** 2 + (y - cpix) ** 2))
    delta_x = 2
    delta_y = -3
    img_x = np.exp(-0.01 * ((x - cpix + delta_x) ** 2 + (y - cpix) ** 2))
    img_y = np.exp(-0.01 * ((x - cpix) ** 2 + (y - cpix + delta_y) ** 2))
    img_xy = np.exp(-0.01 * ((x - cpix + delta_x) ** 2 + (y - cpix + delta_y) ** 2))
    cimg_x = center_psf(img_x)
    cimg_y = center_psf(img_y)
    cimg_xy = center_psf(img_xy)
    # Shape
    assert_equal(cimg_x.shape, img.shape, errshape)
    assert_equal(cimg_y.shape, img.shape, errshape)
    assert_equal(cimg_xy.shape, img.shape, errshape)
    # Central value
    assert_allclose(cimg_x[cpix, cpix], img.max(), err_msg=errval)
    assert_allclose(cimg_y[cpix, cpix], img.max(), err_msg=errval)
    assert_allclose(cimg_xy[cpix, cpix], img.max(), err_msg=errval)
    # Eliminated borders
    assert_equal(
        cimg_x[0 : abs(delta_x)], np.zeros((abs(delta_x), 2 * cpix + 1)), errout
    )
    assert_equal(
        cimg_y[:, -abs(delta_y) :], np.zeros((2 * cpix + 1, abs(delta_y))), errout
    )
    assert_equal(
        cimg_xy[0 : abs(delta_x), -abs(delta_y) :],
        np.zeros((abs(delta_x), abs(delta_y))),
        errout,
    )


def test_circularize(tgrid):
    x, y, cpix = tgrid
    # far from circular
    x1 = cpix // 2
    x2 = -1 - x1
    img1 = x + y
    cimg1 = circularize(img1)
    assert_approx_equal(cimg1[x1, x1], cimg1[x2, x2], 4, errval)
    assert_approx_equal(cimg1[x1, x1], cimg1[x1, x2], 4, errval)
    assert_approx_equal(cimg1[x1, x1], cimg1[x2, x1], 4, errval)
    # quite circular
    img2 = np.exp(-0.01 * ((x - cpix) ** 2 + (y - cpix) ** 2))
    img2[cpix + 3, cpix - 4] = 0.5
    img2[cpix - 2, cpix + 6] = 0.5
    cimg2 = circularize(img2)
    assert_approx_equal(cimg2[x1, x1], cimg2[x2, x2], 4, errval)
    assert_approx_equal(cimg2[x1, x1], cimg2[x1, x2], 4, errval)
    assert_approx_equal(cimg2[x1, x1], cimg2[x2, x1], 4, errval)
    with pytest.raises(AssertionError):
        circularize(img1, log=True)
        circularize(img2, log=True)


def test_center_distance():
    assert_equal(center_distance(15).shape, (15, 15), errshape)
    assert_equal(center_distance(15, 11).shape, (15, 11), errshape)
    with pytest.raises(AssertionError):
        center_distance(-5)
        center_distance(0)
        center_distance(4)
        center_distance(5, 4)
