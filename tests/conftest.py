import numpy as np
import pytest


@pytest.fixture(scope="function", params=[15, 55, 255])
def tones(request):
    size = request.param
    return np.ones((size, size)), size


@pytest.fixture(scope="function", params=[25, 55, 255])
def tgrid(request):
    size = request.param
    x, y = np.mgrid[:size, :size]
    return x, y, size // 2


@pytest.fixture(scope="function", params=[0.11, 0.22, 0.33])
def trange(request):
    step = request.param
    return np.arange(0, 10, step)


@pytest.fixture
def npdata1d():
    return np.arange(10)


@pytest.fixture
def npdata2d():
    return np.arange(25).reshape(5, 5)


@pytest.fixture
def npdata3d():
    return np.arange(64).reshape(4, 4, 4)


@pytest.fixture
def npdata4d():
    return np.arange(81).reshape(3, 3, 3, 3)
