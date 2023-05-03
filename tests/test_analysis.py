from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_approx_equal,
    assert_equal,
)

from psftools.utils.analysis import get_radius, get_radius_old

errshape = "incorrect shape"
errout = "incorrect output"
errval = "incorrect value"


def test_get_radius_old(trange):
    x = trange
    # inside range
    assert_approx_equal(get_radius_old(9, x, x**2), 3.0, 3, errval)
    assert_approx_equal(get_radius_old(125, x, x**3), 5.0, 3, errval)
    assert_approx_equal(get_radius_old(272, x, x**2 + x**4), 4.0, 3, errval)
    # outside range
    assert_equal(get_radius_old(-1, x, x**2), x.max(), errval)
    assert_equal(get_radius_old(100, x, x**2), x.min(), errval)


def test_get_radius(trange):
    x = trange
    # inside range
    assert_approx_equal(get_radius(9, x, x**2), 3.0, 3, errval)
    assert_approx_equal(get_radius(125, x, x**3), 5.0, 3, errval)
    assert_approx_equal(get_radius(272, x, x**2 + x**4), 4.0, 3, errval)
    # outside range
    assert_equal(get_radius(-1, x, x**2), x.max(), errval)
    assert_equal(get_radius(100, x, x**2), x.min(), errval)
