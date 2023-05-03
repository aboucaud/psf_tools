"""
Info submodule
--------------
Gather useful information on a PSF file

"""

import os

from psftools.utils.fits import load_fits_with_header
from psftools.utils.fitting import AiryFit, GaussianFit
from psftools.utils.image import get_e1e2

__all__ = ["infoPSF", "writeInfoPSF", "to_latex"]


def infoPSF(fitcls, image, psc=None, rotation=True, norm_dist=-1):
    info = {}
    if psc:
        info["scale"] = "arcsec"
        mult = psc
        if norm_dist != -1:
            norm_pix = min(norm_dist / psc, image.shape[0] / 2)
        else:
            norm_pix = norm_dist
    else:
        info["scale"] = "pixel"
        mult = 1.0
        norm_pix = -1
    fit2d = fitcls(image, rotation=rotation)
    info["center"] = fit2d.get_center()
    info["fwhm2d"] = fit2d.get_fwhm() * mult
    info["fwhmavg"] = fit2d.get_averaged_fwhm() * mult
    info["r65"] = fit2d.ee2radius(0.65, norm_pix=norm_pix) * mult
    info["r83"] = fit2d.ee2radius(0.83, norm_pix=norm_pix) * mult
    info["r95"] = fit2d.ee2radius(0.95, norm_pix=norm_pix) * mult
    info["ee3x"] = fit2d.radius2ee(3, norm_pix=norm_pix)
    e1, e2 = get_e1e2(image)
    info["e1"] = e1
    info["e2"] = e2
    if rotation:
        info["angle"] = fit2d.get_angle()
        info["eccentricity"] = fit2d.get_eccentricity()
        info["ellipticity"] = fit2d.get_ellipticity()

    # 1D profiles
    xmax, ymax = map(round, fit2d.get_center())
    im1 = image[xmax, :]
    im2 = image[:, ymax]
    for im, axis in zip([im1, im2], ["x", "y"]):
        fit = fitcls(im)
        info[f"fwhm{axis}"] = fit.get_fwhm() * mult
        if fitcls == AiryFit:
            n, p = fit.get_diff_firstlobe()
            info[f"negdif{axis}"] = n
            info[f"posdif{axis}"] = p

    return info


def writeInfoPSF(filename, profil="airy", outpath="."):
    image, hdr = load_fits_with_header(filename)
    if hdr.get("CD1_1"):
        psc = hdr["CD1_1"] * 3600
    else:
        psc = None
    basename = os.path.basename(filename)
    strname = os.path.splitext(basename)[0]
    outtext = """filename\t{fname}
scale   \t{scale}
center_x\t{center[0]:1.3f}
center_y\t{center[1]:1.3f}
fwhm_x  \t{fwhmx:1.3f}
fwhm_y  \t{fwhmy:1.3f}
fwhm_min\t{fwhm2d[0]:1.3f}
fwhm_max\t{fwhm2d[1]:1.3f}
fwhm_avg\t{fwhmavg:1.3f}
r65     \t{r65:1.3f}
r83     \t{r83:1.3f}
r95     \t{r95:1.3f}
ee3x    \t{ee3x:1.3f}
eccent  \t{eccentricity:1.3f}
ellipt  \t{ellipticity:1.3f}
"""

    if profil.lower() == "gaussian":
        clas = GaussianFit
    elif profil.lower() == "airy":
        clas = AiryFit
    else:
        raise NotImplementedError()
    header = f"# Results of the {profil.capitalize} fit:\n"
    outname = f"{outpath}/{strname}_{profil.lower()}.txt"
    with open(outname, "w") as outfile:
        dico = infoPSF(clas, image, psc=psc, rotation=True)
        outfile.write(header)
        outfile.write(outtext.format(fname=basename, **dico))
    print(f"{profil.capitalize()} fit done in {os.path.basename(outname)}")


def to_latex(a, label="A \\oplus B"):
    import sys

    sys.stdout.write(
        "\\[ {} = \\left| \\begin{{array}}{{{}}}\n".format(label, "c" * a.shape[1])
    )
    for r in a:
        sys.stdout.write(" & ".join(r))
        # sys.stdout.write(str(r[0]))
        # for c in r[1:]:
        #    sys.stdout.write(' & ' + str(c))
        sys.stdout.write("\\\\\n")
    sys.stdout.write("\\end{array} \\right| \\]\n")
