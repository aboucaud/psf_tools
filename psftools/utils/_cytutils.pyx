cimport cython
from libc.math cimport NAN, isnan, pow, sqrt

import numpy as np

cimport numpy as np

FTYPE = np.float64
ctypedef np.float64_t FTYPE_t
ITYPE = np.int64
ctypedef np.int64_t ITYPE_t

@cython.boundscheck(False)
cdef inline FTYPE_t distance(int x, int y, double xmid, double ymid):
    return sqrt(pow(x-xmid, 2) + pow(y-ymid, 2))

# @cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def azimuthal_profile(np.ndarray[FTYPE_t, ndim=2] image,
                      np.ndarray[FTYPE_t, ndim=1] origin,
                      FTYPE_t bin,
                      int nbins):

    assert image.dtype == FTYPE

    cdef unsigned int nx = image.shape[0]
    cdef unsigned int ny = image.shape[1]
    cdef double xmid = origin[0]
    cdef double ymid = origin[1]

    cdef unsigned int i, j, ibin
    cdef FTYPE_t dist, val

    cdef np.ndarray[FTYPE_t, ndim=1] rad = np.zeros(nbins, dtype=FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] profile = np.zeros(nbins, dtype=FTYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] histo = np.zeros(nbins, dtype=ITYPE)

    for i in range(nx):
        for j in range(ny):
            val = image[i,j]
            if isnan(val):
                continue
            dist = distance(i, j, xmid, ymid)
            ibin = <int>(dist / bin)
            if ibin >= nbins:
                continue
            rad[ibin] += dist
            profile[ibin] += val
            histo[ibin] += 1

    for i in range(nbins):
        if histo[i] != 0:
            rad[i] /= histo[i]
            profile[i] /= histo[i]
        else:
            rad[i] = bin * (i + 0.5)
            profile[i] = NAN

    return rad, profile, histo


@cython.boundscheck(False)
@cython.wraparound(False)
def get_radius(FTYPE_t value,
               np.ndarray[FTYPE_t, ndim=1] dist,
               np.ndarray[FTYPE_t, ndim=1] profile):

    assert dist.size == profile.size

    cdef unsigned int idx, idmin, idmax
    cdef FTYPE_t radius, pmax, pmin, dmin, dmax, pivot, delta

    cdef np.ndarray[FTYPE_t, ndim=1] scaled_profile = np.zeros(dist.size,
                                                               dtype=FTYPE)

    pmax = profile.max()
    pmin = profile.min()

    if value >= pmax:
        radius = dist.min()
    elif value <= pmin:
        radius = dist.max()
    else:
        scaled_profile = profile - value
        idx = np.argmin(np.abs(scaled_profile))
        pivot = scaled_profile[idx]
        if pivot < 0:
            idmin = idx - 1
            idmax = idx
        else:
            idmin = idx
            idmax = idx + 1
        delta = (value - profile[idmin]) / (profile[idmax] - profile[idmin])
        radius = dist[idmin] + delta * (dist[idmax] - dist[idmin])

    return radius
