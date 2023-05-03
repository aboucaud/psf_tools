"""
Plotting submodule
------------------
Useful plotting methods

"""
import numpy as np
from astropy.visualization import LogStretch, simple_norm

__all__ = ["plot_image", "plot_log", "plot_stretch"]


def plot_image(image, filepath, log=False, tridim=False, cmap="coolwarm"):
    """
    Creates plot of the current image

    Parameters
    ----------
    image : real `numpy.ndarray`
        Image to plot
    filepath : str
        Filename with full path
    log : bool, optional
        If `True`, plot the log scaled intensity.
        Only works for 2D plots. (default `False`)
    tridim : bool, optional
        If `True`, renders as 3d instead of 2d (default `False`)
    cmap : str, optional
        Colormap for the plot
            * 'gray'
            * 'jet'
            * 'coolwarm' (default)

    """
    import matplotlib.pyplot as plt

    # Normalization to the max value
    image /= image.max()

    # PLOT
    fig = plt.figure()
    if tridim:
        from mpl_toolkits.mplot3d import axes3d  # noqa

        sxy = image.shape[0]
        x, y = np.mgrid[:sxy, :sxy] - sxy // 2
        xydev = round(0.1 * sxy)
        zdev = 0.2
        ax = fig.add_subplot(111, projection="3d")
        plot = ax.plot_surface(
            x, y, image, rstride=1, cstride=1, cmap=cmap, linewidth=0, alpha=0.5
        )

        ax.contour(x, y, image, zdir="x", offset=x.min() - xydev, cmap=cmap)
        ax.contour(x, y, image, zdir="y", offset=y.max() + xydev, cmap=cmap)
        ax.contour(x, y, image, zdir="z", offset=-zdev, cmap=cmap, alpha=0.7)

        ax.set_xlabel(r"x")
        ax.set_xlim(x.min() - xydev, x.max() + xydev)
        ax.set_ylabel(r"y")
        ax.set_ylim(y.min() - xydev, y.max() + xydev)
        ax.set_zlabel(r"z")
        ax.set_zlim(0 - zdev, 1 + zdev)
    else:
        ax = fig.add_subplot(111)
        if log:
            from matplotlib.colors import LogNorm

            plot = ax.imshow(image, norm=LogNorm(), interpolation="nearest", cmap=cmap)
        else:
            plot = ax.imshow(image, interpolation="nearest", cmap=cmap)
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")

    fig.colorbar(plot, shrink=0.7)
    fig.savefig(filepath, dpi=144)
    plt.close(fig)

    # print("Image has been saved as {}".format(filepath))


def plot_log(img, a=1000, cmap="CMRmap"):
    """
    Display the logarithmically stretched image

    Parameters
    ----------
    img : array_like
        image to display
    a : float, optional
        stretch factor (default 1000)
    cmap : str, optional
        colormap for the plot (any matplotlib colormap)
        (default 'CMRmap')

    """
    import matplotlib.pyplot as plt

    logstretched = LogStretch(a=a)(img.copy())
    plt.figure(figsize=(12, 12))
    plt.imshow(logstretched, cmap=cmap)
    plt.axis("off")
    plt.show()


def plot_stretch(img, scale="log", cmap="Greys_r", **kwargs):
    """
    Display the stretched image using a given scale

    Parameters
    ----------
    img : array_like
        image to display
    scale : str, optional
        stretch scale
            * 'linear'
            * 'log' (default)
            * 'sqrt'
            * 'asinh'
    cmap : str, optional
        colormap for the plot (any matplotlib colormap)
        (default 'Greys_r')

    """
    import matplotlib.pyplot as plt

    norm = simple_norm(img, stretch=scale)
    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap=cmap, norm=norm, **kwargs)
    plt.axis("off")
    plt.show()
