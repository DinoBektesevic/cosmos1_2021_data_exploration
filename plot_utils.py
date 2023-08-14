import numpy as np
import astropy.units as u
from astropy.wcs import WCS

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
    
    return xy, width, height, angle
        
    
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


def plot_focal_plane(ax, hdulist, showExtName=True, txtOffset=30*u.arcsec, figure=None):
    """Plots the footprint of given HDUList on the axis."""
    wcss = [WCS(hdu.header) for hdu in hdulist]
    # I really wish that WCS would pop an error when unable to 
    # init from a header instead of defaulting, oh, and __eq__ 
    # doesn't compare naturally
    default_wcs = WCS().to_header_string()
    for hdu in hdulist:
        wcs = WCS(hdu.header)
        if default_wcs != wcs.to_header_string():
            ax = plot_footprint(ax, wcs)
            pt = wcs.pixel_to_world(0, 0)
            x, y = pt.ra.deg+0.01, pt.dec.deg+0.01
            ax.text(x, y, hdu.header.get("EXTNAME", None))
        
    #ax = plot_footprints(ax, wcss)
    return ax