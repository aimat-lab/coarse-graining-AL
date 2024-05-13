from warnings import warn as _warn
from pyemma.plots.plots2d import _get_cmap
from pyemma.plots.plots2d import get_histogram
from pyemma.plots.plots2d import plot_map
from pyemma.plots.plots2d import _to_free_energy
import numpy as _np
import matplotlib.pyplot as plt

# This function has been copied from the pyemma package and slightly modified.


def plot_free_energy_histogram(
    xall,
    yall,
    weights=None,
    ax=None,
    nbins=100,
    ncontours=100,
    offset=-1,
    avoid_zero_count=False,
    minener_zero=True,
    kT=1.0,
    vmin=None,
    vmax=None,
    cmap="nipy_spectral",
    cbar=True,
    cbar_label=r"free energy / $kT$",
    cax=None,
    levels=None,
    legacy=True,
    ncountours=None,
    cbar_orientation="vertical",
    **kwargs
):
    """Plot a two-dimensional free energy map using a histogram of
    scattered data.

    Parameters
    ----------
    xall : ndarray(T)
        Sample x-coordinates.
    yall : ndarray(T)
        Sample y-coordinates.
    weights : ndarray(T), optional, default=None
        Sample weights; by default all samples have the same weight.
    ax : matplotlib.Axes object, optional, default=None
        The ax to plot to; if ax=None, a new ax (and fig) is created.
        Number of contour levels.
    nbins : int, optional, default=100
        Number of histogram bins used in each dimension.
    ncontours : int, optional, default=100
        Number of contour levels.
    offset : float, optional, default=-1
        Deprecated and ineffective; raises a ValueError
        outside legacy mode.
    avoid_zero_count : bool, optional, default=False
        Avoid zero counts by lifting all histogram elements to the
        minimum value before computing the free energy. If False,
        zero histogram counts would yield infinity in the free energy.
    minener_zero : boolean, optional, default=True
        Shifts the energy minimum to zero.
    kT : float, optional, default=1.0
        The value of kT in the desired energy unit. By default,
        energies are computed in kT (setting 1.0). If you want to
        measure the energy in kJ/mol at 298 K, use kT=2.479 and
        change the cbar_label accordingly.
    vmin : float, optional, default=None
        Lowest free energy value to be plotted.
        (default=0.0 in legacy mode)
    vmax : float, optional, default=None
        Highest free energy value to be plotted.
    cmap : matplotlib colormap, optional, default='nipy_spectral'
        The color map to use.
    cbar : boolean, optional, default=True
        Plot a color bar.
    cbar_label : str, optional, default='free energy / kT'
        Colorbar label string; use None to suppress it.
    cax : matplotlib.Axes object, optional, default=None
        Plot the colorbar into a custom axes object instead of
        stealing space from ax.
    levels : iterable of float, optional, default=None
        Contour levels to plot.
    legacy : boolean, optional, default=True
        Switch to use the function in legacy mode (deprecated).
    ncountours : int, optional, default=None
        Legacy parameter (typo) for number of contour levels.
    cbar_orientation : str, optional, default='vertical'
        Colorbar orientation; choose 'vertical' or 'horizontal'.

    Optional parameters for contourf (**kwargs)
    -------------------------------------------
    corner_mask : boolean, optional
        Enable/disable corner masking, which only has an effect if
        z is a masked array. If False, any quad touching a masked
        point is masked out. If True, only the triangular corners
        of quads nearest those points are always masked out, other
        triangular corners comprising three unmasked points are
        contoured as usual.
        Defaults to rcParams['contour.corner_mask'], which
        defaults to True.
    alpha : float
        The alpha blending value.
    locator : [ None | ticker.Locator subclass ]
        If locator is None, the default MaxNLocator is used. The
        locator is used to determine the contour levels if they are
        not given explicitly via the levels argument.
    extend : [ ‘neither’ | ‘both’ | ‘min’ | ‘max’ ]
        Unless this is ‘neither’, contour levels are automatically
        added to one or both ends of the range so that all data are
        included. These added ranges are then mapped to the special
        colormap values which default to the ends of the
        colormap range, but can be set via
        matplotlib.colors.Colormap.set_under() and
        matplotlib.colors.Colormap.set_over() methods.
    xunits, yunits : [ None | registered units ]
        Override axis units by specifying an instance of a
        matplotlib.units.ConversionInterface.
    antialiased : boolean, optional
        Enable antialiasing, overriding the defaults. For filled
        contours, the default is True. For line contours, it is
        taken from rcParams[‘lines.antialiased’].
    nchunk : [ 0 | integer ]
        If 0, no subdivision of the domain. Specify a positive
        integer to divide the domain into subdomains of nchunk by
        nchunk quads. Chunking reduces the maximum length of polygons
        generated by the contouring algorithm which reduces the
        rendering workload passed on to the backend and also requires
        slightly less RAM. It can however introduce rendering
        artifacts at chunk boundaries depending on the backend, the
        antialiased flag and value of alpha.
    hatches :
        A list of cross hatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour. Hatching
        is supported in the PostScript, PDF, SVG and Agg backends
        only.
    zorder : float
        Set the zorder for the artist. Artists with lower zorder
        values are drawn first.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    misc : dict
        Contains a matplotlib.contour.QuadContourSet 'mappable' and,
        if requested, a matplotlib.Colorbar object 'cbar'.

    """
    if legacy:
        _warn(
            "Legacy mode is deprecated is will be removed in the"
            " next major release. Until then use legacy=False",
            DeprecationWarning,
        )
        cmap = _get_cmap(cmap)
        if offset != -1:
            _warn(
                "Parameter offset is deprecated and will be ignored", DeprecationWarning
            )
        if ncountours is not None:
            _warn(
                "Parameter ncountours is deprecated;" " use ncontours instead",
                DeprecationWarning,
            )
            ncontours = ncountours
        if vmin is None:
            vmin = 0.0
    else:
        if offset != -1:
            raise ValueError("Parameter offset is not allowed outside legacy mode")
        if ncountours is not None:
            raise ValueError(
                "Parameter ncountours is not allowed outside"
                " legacy mode; use ncontours instead"
            )
    x, y, z = get_histogram(
        xall, yall, nbins=nbins, weights=weights, avoid_zero_count=avoid_zero_count
    )
    f = _to_free_energy(z, minener_zero=minener_zero) * kT

    if vmax is not None:  # Modification from original code
        f[f >= vmax] = _np.inf

    if False:
        fig, ax, misc = plot_map(
            x,
            y,
            f,
            ax=ax,
            cmap=cmap,
            ncontours=ncontours,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cbar=cbar,
            cax=cax,
            cbar_label=cbar_label,
            cbar_orientation=cbar_orientation,
            norm=None,
            **kwargs
        )

    else:
        if ax is None:
            fix = plt.figure()
            ax = fix.gca()
        else:
            fig = ax.get_figure()

        if True:
            map = ax.imshow(
                # f.T,
                f,
                extent=[
                    x.min(),
                    x.max(),
                    y.min(),
                    y.max(),
                ],
                origin="lower",
                cmap="nipy_spectral",
                interpolation="nearest",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            map = ax.imshow(
                _np.exp(-f) * _np.exp(-_np.flip(f, axis=(0, 1))),
                extent=[
                    x.min(),
                    x.max(),
                    y.min(),
                    y.max(),
                ],
                origin="lower",
                cmap="nipy_spectral",
                interpolation="nearest",
                aspect="auto",
                # vmin=vmin,
                # vmax=vmax,
            )

        if cbar:
            if cax is None:
                cbar_ = fig.colorbar(map, ax=ax, orientation=cbar_orientation)
            else:
                cbar_ = fig.colorbar(map, cax=cax, orientation=cbar_orientation)
            if cbar_label is not None:
                cbar_.set_label(cbar_label)

    return fig, ax


def plot_free_energy(
    free_energies,
    phi_dih,
    psi_dih,
    save_path=None,
    vmin=0.0,
    vmax=11.0,
    colorbar_label=None,
):
    fig = plt.figure()
    plt.scatter(
        phi_dih,
        psi_dih,
        c=free_energies,
        s=1,
        cmap="nipy_spectral",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("phi")
    plt.ylabel("psi")
    plt.colorbar(label=colorbar_label)

    if save_path is not None:
        plt.savefig(save_path)

    return fig
