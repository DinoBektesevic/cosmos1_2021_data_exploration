import warnings

import numpy as np

import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import simple_norm, ZScaleInterval, AsinhStretch, ImageNormalize

import matplotlib.pyplot as plt


def iter_over_obj(objects):
    """Folds the given list of objects on their ``Name``s and
    iterates over them sorted by date-time stamp.
                
    Parameters
    -----------
    objects : `astropy.table.Table`
        Table of objects.
        
    Returns
    --------
    obj : `iterator`
        Iterator over individual object observations.
    """
    names = set(objects["Name"])
    for name in names:
        obj = objects[objects["Name"] == name]
        obj.sort("obstime")
        yield obj

        
def transform_rect(points):
    """Given a rectangle defined by 4 points (clockwise convention)
    returns top-left point, width, height, and angle of rectangle.
    
    Parameters
    ----------
    points : `list`
        List of 4 tuples representing (x, y) coordinates of the
        corners of a rectanlge, in clockwise convention.
        
    Returns
    -------
    xy : `tuple`
        Top left corner (x, y) coordinates.
    width : `float`
        Width
    height : `float`
        Height
    angle : `float`
        Angle of rotation, in radians.
    """
    calc_dist = lambda p1, p2: np.sqrt( (p1[0] - p2[0])**2 + ( p1[1] - p2[1])**2 )
    calc_angle = lambda p1, p2: np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    
    # flip height so the xy becomes top left, then we don't have to guess
    # which point we need to return
    width = calc_dist(points[0], points[1])
    height = calc_dist(points[1], points[2])
    xy = points[0]
    
    angle = calc_angle(points[-1], points[0])
    
    return xy, width, -height, angle
        
    
def plot_field(ax, center, radius, figure=None):
    """Adds a circle of the given radius and the origin at the given
    center coordinates to the given axis."""
    ax.scatter(*center, color="black", label="Pointing area")
    circ = plt.Circle(center, radius, fill=False, color="black")
    ax.add_artist(circ)
    return ax
    
    
def plot_footprint(ax, wcs, figure=None):
    """Adds the footprint defined by the given WCS to the axis."""
    xy, width, height, angle = transform_rect(wcs.calc_footprint())
    rect = plt.Rectangle(xy, width, height, angle=angle, fill=None, color="black")
    ax.add_artist(rect)
    return ax
    
    
def plot_footprints(ax, wcs_list, figure=None):
    """Adds the footprints defined by each given WCS to the axis."""
    for wcs in wcs_list:
        ax = plot_footprint(ax, wcs, figure)
    return ax
    
    
def plot_all_objs(ax, objects, center, count=0, show_field=False,
                  radius=1.1, lw=0.9, ms=1, figure=None):
    """Plots object tracks on the given axis.
    
    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    objects : `astropy.table.Table`
        Table of objects.
    count : `int`
        Number of tracks to plot.
    show_field : `bool`
        Show approximate COSMOS 1 field.
        
    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    """
    if show_field:
        plot_field(ax, center, radius)
        
    for i, obj in enumerate(iter_over_obj(objects)):
        if count > 0 and i == count:
            break
        ax.plot(obj["RA"], obj["DEC"], label=obj["Name"][0], marker="o", lw=lw, ms=ms)
        
    if figure is not None:
        return ax, figure
    return ax


def plot_focal_plane(ax, hdulist, showExtName=True, txt_x_offset=20*u.arcsec, txt_y_offset=-120*u.arcsec, figure=None):
    """Plots the footprint of given HDUList on the axis."""
    with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            wcss = [WCS(hdu.header) for hdu in hdulist]
    # I really wish that WCS would pop an error when unable to 
    # init from a header instead of defaulting, oh, and __eq__ 
    # doesn't compare naturally
    default_wcs = WCS().to_header_string()
    for hdu in hdulist:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            wcs = WCS(hdu.header)
            
        if default_wcs != wcs.to_header_string():
            ax = plot_footprint(ax, wcs)
            pt = wcs.pixel_to_world(0, 0)
            # I swear to got god damn units vs quantities
            xoffset = txt_x_offset.to(u.deg).value
            yoffset = txt_y_offset.to(u.deg).value
            # we need to move diagonally to the right and down to center the text
            x, y = pt.ra.deg+xoffset , pt.dec.deg+yoffset
            ax.text(x, y, hdu.header.get("EXTNAME", None), clip_on=True)
        
    #ax = plot_footprints(ax, wcss)
    return ax


def plot_cutouts(axes, cutouts, remove_extra_axes=True):
    """Plots cutouts (images) onto given axes. 
    
    The number of axes must be equal to or greater
    than the number of cutouts.
    
    Parameters
    ----------
    ax : `list[matplotlib.pyplot.Axes]`
        Axes.
    cutouts : `list`, `np.array` or `astropy.ndutils.Cutout2D`
        Collection of numpy arrays or ``Cutout2D`` objects 
        to plot.
    remove_extra_axes : `bool`, optional
         When `True` (default), the axes that would be 
         left empty are removed from the plot. 
         
    Raises
    -------
    ValueError - When number of given axes is less than
    the number of given cutouts.
    """ 
    nplots = len(cutouts)
    
    axs = axes.ravel()
    naxes = len(axs)
    
    if naxes < nplots:
        raise ValueError(f"N axes ({len(axes)}) doesn't match N plots ({nplots}).")

    for ax, cutout in zip(axs, cutouts):
        img = cutout.data if isinstance(cutout, Cutout2D) else cutout
        norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
        im = ax.imshow(cutout.data, norm=norm)
        ax.set_aspect("equal")
        ax.axvline(cutout.shape[0]/2, c="red", lw=0.25)
        ax.axhline(cutout.shape[1]/2, c="red", lw=0.25)
        
    
    if remove_extra_axes and naxes > nplots:
        for ax in axs[nplots-naxes:]:
            ax.remove()        

    return axes


def plot_img(img, ax=None, norm=True, title=None):
    """Plots an image on an axis, if no axis is given
    creates a new figure. Draws a crosshair at the 
    center of the image.
    
    Parameters
    ----------
    img : `np.array`
        Image array
    ax : `matplotlib.pyplot.Axes` or `None`
        Axes, `None` by default.
    norm: `bool`, optional
        Normalize the image using Astropy's `ImageNormalize`
        using `ZScaleInterval` and `AsinhStretch`. `True` by
        default.
    title : `str` or None, optional
        Title of the plot.     
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
     
    if norm:
        norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
        im = ax.imshow(img, norm=norm)
    else:
        im = ax.imshow(img)
        
    ax.axvline(img.shape[0]/2, c="red", lw=0.5)
    ax.axhline(img.shape[1]/2, c="red", lw=0.5)
    ax.set_title(title)
    fig.colorbar(im, label="Counts")
    
    return fig, ax