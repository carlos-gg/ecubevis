from typing import Type
import matplotlib
import numpy as np
from numpy.core.fromnumeric import var
import xarray as xr
import hvplot.xarray 
import cartopy.crs as crs
import holoviews as hv
import matplotlib.colors as colors
from matplotlib.pyplot import figure, show, savefig, close, subplots, Axes
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .io import load_transform_mfdataset
from .utils import check_coords, slice_dataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = ['plot_dataset',
           'plot_ndarray',
           'cm', 
           'crs']


def _bold(string):
   return '\033[1m' + string + '\033[0m'


def plot_ndarray(
    data, 
    interactive=True, 
    colorbar=True, 
    cmap='viridis', 
    share_dynamic_range=True,
    vmin=None, 
    vmax=None, 
    dpi=80,
    coastline=False,
    max_static_subplot_rows=10,
    max_static_subplot_cols=10):
    """
    Plot a 3D or 4D `numpy` array or a tuple of 2D `numpy` arrays. 
    
    Parameters
    ----------
    interactive : bool, optional
        Whether to plot using an interactive (with ``bokeh``) or static (with
        ``matplotlib``) plot. In the case of a 3D ndarray, a slider will be used 
        to explore the data across the 1 dimension (time or vertical level). In 
        the case of a 3D ndarray, a slider will be used to explore the data 
        across the 1 and 2 dimensions. 
    """
    if interactive:
        pass
    else:
        if isinstance(data, tuple):
            if isinstance(data[0], np.ndarray) and data[0].ndim == 2:
                data = np.concatenate([np.expand_dims(im, 0) for im in data], 
                                      axis=0)
                mosaic_orientation = 'row'
            else:
                raise TypeError('wrong data tuple')
        else:
            mosaic_orientation = 'col'

        if share_dynamic_range:
            if vmin is None:
                vmin = data.min()
                vmin = np.array(vmin)
            if vmax is None:
                vmax = data.max() 
                vmax = np.array(vmax)
        
        return _plot_mosaic_3or4d(data, 
                                  show_colorbar=colorbar, 
                                  dpi=dpi, 
                                  cmap=cmap, 
                                  show_axis=True, 
                                  save=None, 
                                  vmin=vmin, 
                                  vmax=vmax, 
                                  transparent=False, 
                                  coastline=coastline, 
                                  mosaic_orientation=mosaic_orientation,
                                  )


def plot_dataset(
    data, 
    interactive=True, 
    variable=None, 
    groupby=None, 
    slice_time=None, 
    slice_level=None, 
    slice_lat=None, 
    slice_lon=None, 
    colorbar=True, 
    cmap='viridis', 
    logz=False, 
    share_dynamic_range=True, 
    vmin=None, 
    vmax=None, 
    wanted_projection=None, 
    data_projection=crs.PlateCarree(),
    coastline=False, 
    global_extent=False, 
    extent=None,
    dynamic=True, 
    dpi=80, 
    max_static_subplot_rows=10,
    max_static_subplot_cols=10,
    plot_sizepx=1000, 
    widget_location='top', 
    verbose=True):
    """
    Plot an n-dimensional dataset (in-memory or from a path). The dataset is 
    loaded through ``xarray`` and therefore supports formats such as NetCDF, 
    IRIS or GRIB.

    Parameters
    ----------
    data : xarray Dataset/Dataarray or str
        ERA5 variable(s) as Xarray (in memory) variable or as a string with 
        the path to the corresponding NetCDF file. Expected dimensions: 
        4D array [time, level, lat, lon] or 3D array [time, lat, lon].
    interactive : bool optional
        Whether to plot using an interactive plot (using ``hvplot``) with a 
        slider across the dimension set by ``groupby`` or an static mosaic 
        (using ``matplotlib``). 
    variable : str or int or None, optional
        The name of the variable to be plotted or the index at which it is 
        located. If None, the first 3D or 4D variable is selected.
    slice_time : tuple of int or str or None, optional
        Tuple with initial and final values for slicing the time dimension. If 
        None, the array is not sliced accross this dimension.
    slice_level : tuple of int or None, optional
        Tuple with initial and final values for slicing the level dimension. If 
        None, the array is not sliced accross this dimension.
    slice_lat : tuple of int or None, optional
        Tuple with initial and final values for slicing the lat dimension. If 
        None, the array is not sliced accross this dimension.
    slice_lon : tuple of int or None, optional
        Tuple with initial and final values for slicing the lon dimension. If 
        None, the array is not sliced accross this dimension.
    colorbar : bool optional
        To show a colorbar.
    cmap : str or matplotlib.cm, optional
        Colormap, eg. viridis" or ecv.cm.viridis.  
    projection : cartopy.crs projection, optional
        According to Cartopy's documentation it can be one of the following
        (https://scitools.org.uk/cartopy/docs/latest/crs/projections.html): 
        PlateCarree, AlbersEqualArea, AzimuthalEquidistant, EquidistantConic, 
        LambertConformal, LambertCylindrical, Mercator, Miller, Mollweide, 
        Orthographic, Robinson, Sinusoidal, Stereographic, TransverseMercator, 
        UTM, InterruptedGoodeHomolosine, RotatedPole, OSGB, EuroPP, Geostationary, 
        NearsidePerspective, EckertI, EckertII, EckertIII, EckertIV, EckertV, 
        EckertVI, EqualEarth, Gnomonic, LambertAzimuthalEqualArea, 
        NorthPolarStereo, OSNI, SouthPolarStereo. Can be called as 
        ``ecv.crs.PlateCarree()``.
    extent : tuple of 4 floats
        A tuple with four values in the format (lon_ini, lon_fin, lat_ini, 
        lat_fin). Used to zoom the map to a given bounding box. Valid for static 
        plots, when coastline is shown. 

    Notes
    -----
    https://github.com/pydata/xarray/issues/2199
    https://hvplot.holoviz.org/user_guide/Gridded_Data.html
    https://hvplot.holoviz.org/user_guide/Geographic_Data.html    

    TODO
    ----
    [1]
    for hvplot: col='time'
    https://hvplot.holoviz.org/user_guide/Subplots.html

    [2]
    https://pyviz-dev.github.io/holoviz/tutorial/Composing_Plots.html

    """     
    if isinstance(data, str):
        if not data.endswith('.nc'):
            data += '.nc'
        data = xr.open_dataset(data, engine="netcdf4", decode_times=True, 
                               chunks={'time': 1}) 
    
    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError('`data` must be an Xarray Dataset/Dataarray')  
    
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()

    ### Selecting the variable 
    if variable is None: # taking the first 3D or 4D data variable
        for i in data.data_vars:
            if data.data_vars.__getitem__(i).ndim >= 3:
                variable = i
    elif isinstance(variable, int):
        variable = list(data.keys())[variable]
    else: # otherwise it is the variable name as a string
        if not isinstance(variable, str):
            raise ValueError('`variable` must be None, int or str')
    
    ### Getting info
    shape = data.data_vars.__getitem__(variable).shape
    tini = data.data_vars.__getitem__(variable).time[0].values
    tini = np.datetime_as_string(tini, unit='m')
    tfin = data.data_vars.__getitem__(variable).time[-1].values
    tfin = np.datetime_as_string(tfin, unit='m')
    var_array = check_coords(data)
    
    ### Slicing the array variable
    if not interactive:
        if slice_time is None and 'time' in var_array.coords and \
            var_array.time.size > max_static_subplot_rows:
            if verbose:
                print(f'Showing the first {max_static_subplot_rows} time steps '
                      'according to `max_static_subplot_rows` argument \n')
            slice_time = (0, max_static_subplot_rows) 
        if slice_level is None and 'level' in var_array.coords and \
            var_array.level.size > max_static_subplot_cols:
            if verbose:
                print(f'Showing the first {max_static_subplot_cols} level steps '
                      'according to `max_static_subplot_cols` argument \n')
            slice_level = (0, max_static_subplot_cols) 
    var_array = slice_dataset(var_array, slice_time, slice_level, slice_lat, 
                              slice_lon)  
    var_array = var_array.data_vars.__getitem__(variable)
    
    if groupby is None:
        if interactive:
            groupby = ['time', 'level'] if var_array.ndim == 4 else 'time'
        else:
            groupby = 'time'
    
    if not var_array.ndim in [3, 4]:
        raise TypeError('Variable is neither 3D nor 4D')
 
    if verbose in [1, 2]:
        shape_slice = var_array.shape
        # assuming the min temporal sampling unit is minutes
        tini_slice = np.datetime_as_string(var_array.time[0].values, unit='m')
        tfin_slice = np.datetime_as_string(var_array.time[-1].values, unit='m')
        dimp = '4D' if var_array.ndim == 4 else '3D'
        if hasattr(var_array, 'long_name'):
            print(f'{_bold("Name")} {variable}, {var_array.long_name}')
        else:
            print(f'{_bold("Name")} {variable}')
        if hasattr(var_array, 'units'):
            print(f'{_bold("Units:")} {var_array.units}') 
        print(f'{_bold("Dimensionality:")} {dimp}') 
        print(f'{_bold("Shape:")} {shape}')
        print(f'{_bold("Shape (sliced array):")} {shape_slice}')
        print(f'{_bold("Time interval:")} {tini} --> {tfin}')
        print(f'{_bold("Time interval (sliced array):")} {tini_slice} --> {tfin_slice}\n')
    if verbose in [2]:
        print(data.coords)
        print(data.data_vars, '\n')
    
    sizey = var_array.lat.shape[0]
    sizex = var_array.lon.shape[0]
    sizexy_ratio = sizex / sizey

    ### interactive plotting with slider(s) using bokeh
    if interactive:
        hv.extension('bokeh')
        if coastline or wanted_projection is not None:
            width = plot_sizepx
            height = int(np.round(width / sizexy_ratio))
            height += int(height * 0.2)
        else:
            width = plot_sizepx
            height = int(np.round(width / sizexy_ratio))

        sizeargs = dict(height=height, width=width)
        project = False if wanted_projection is None else True
    
        return var_array.hvplot(kind='image', 
                                x='lon', 
                                y='lat', 
                                groupby=groupby, 
                                dynamic=dynamic, 
                                colorbar=colorbar, 
                                cmap=cmap, 
                                shared_axes=True, 
                                legend=True, 
                                logz=logz, 
                                widget_location=widget_location, 
                                project=project, 
                                projection=wanted_projection, 
                                global_extent=global_extent, 
                                coastline=coastline, 
                                **sizeargs)
        
    ### Static mosaic with matplotlib
    else:                
        if share_dynamic_range:
            if vmin is None:
                vmin = var_array.min().compute()
                vmin = np.array(vmin)
            if vmax is None:
                vmax = var_array.max().compute()   
                vmax = np.array(vmax)
        
        return _plot_mosaic_3or4d(var_array, 
                                  show_colorbar=colorbar, 
                                  dpi=dpi, 
                                  cmap=cmap, 
                                  logscale=logz, 
                                  show_axis=True, 
                                  save=None, 
                                  vmin=vmin, 
                                  vmax=vmax, 
                                  transparent=False, 
                                  coastline=coastline, 
                                  wanted_projection=wanted_projection,
                                  data_projection=data_projection,
                                  global_extent=global_extent,
                                  extent=extent)
                

def _plot_mosaic_3or4d(
    data, 
    show_colorbar=True, 
    dpi=100, 
    cmap='viridis', 
    logscale=False, 
    show_axis=True, 
    save=None, 
    vmin=None, 
    vmax=None, 
    transparent=False, 
    coastline=False, 
    wanted_projection=None,
    data_projection=None,
    global_extent=False,
    extent=None,
    mosaic_orientation='col'):
    """
    
    Ticks with non-rectangular projection supported in Carotpy 0.18
    https://scitools.org.uk/cartopy/docs/latest/gallery/gridliner.html
    axis.gridlines(draw_labels=True)

    On Cartopy 0.17
    TypeError: Cannot label gridlines on a EqualEarth plot. 
    Only PlateCarree and Mercator plots are currently supported.

    """
    params = dict()
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        use_xarray = True
    else:
        if not isinstance(data, np.ndarray):
            raise TypeError('data format not supported')
        use_xarray = False

    if use_xarray:
        sizexy_ratio = data.lon.shape[0] / data.lat.shape[0]
        if 'level' in data.coords:
            cols = data.level.shape[0]
        else:
            cols = 1
        rows = data.time.shape[0]
    else:
        if data.ndim == 2:
            sizexy_ratio = data.shape[1] / data.shape[0]
            cols = 1
            rows = 1
        elif data.ndim == 3:
            sizexy_ratio = data.shape[2] / data.shape[1] 
            if mosaic_orientation == 'col':
                cols = 1
                rows = data.shape[0]
            elif mosaic_orientation == 'row':
                rows = 1
                cols = data.shape[0]
        elif data.ndim == 4:
            sizexy_ratio = data.shape[3] / data.shape[2]
            cols = data.shape[1]
            rows = data.shape[0]     

    if use_xarray:
        lon_ini = data.lon[0].values
        lon_fin = data.lon[-1].values
        lat_ini = data.lat[0].values
        lat_fin = data.lat[-1].values
        extent_known = True
    else:
        if extent is not None:
            lon_ini, lon_fin, lat_ini, lat_fin = extent
            extent_known = True
        else:
            extent_known = False

    figcols = cols * 2  
    figrows = rows * 2 
    colorbarzone = 1.4 if show_colorbar else 1
    if mosaic_orientation == 'col':
        figsize = (max(8, figcols) * sizexy_ratio * colorbarzone, max(8, figrows))    
    elif mosaic_orientation == 'row':
         figsize = (max(8, figrows), max(8, figcols) * sizexy_ratio * colorbarzone) 
    fig, ax = subplots(rows, cols, sharex='col', sharey='row', dpi=dpi, 
                       figsize=figsize, constrained_layout=False, 
                       subplot_kw={'projection': wanted_projection})

    data = np.squeeze(data)
    for i in range(rows):
        for j in range(cols):
            if cols == 1:
                if rows == 1:
                    axis = ax
                    image = data
                else:
                    axis = ax[i]
                    image = data[i]
                if use_xarray:
                    time = np.datetime64(image.time.values, 'm')
                    axis.set_title(f'$\ittime$={time}', fontsize=10)
            elif rows == 1:
                axis = ax[j]
                image = data[j]
                if use_xarray:
                    level = image.level.values
                    axis.set_title(f'$\itlevel$={level}', fontsize=10)
            else:
                axis = ax[i, j]
                image = data[i, j]
                time = np.datetime64(image.time.values, 'm')
                if use_xarray:
                    level = image.level.values
                    axis.set_title(f'$\ittime$={time}, $\itlevel$={level}', 
                                   fontsize=10)

            if logscale:
                image += np.abs(image.min())
                if vmin is None:
                    linthresh = 1e-2
                else:
                    linthresh = vmin
                    norm = colors.SymLogNorm(linthresh)   
            else:
                norm = None                

            if coastline:
                axis.coastlines()
                axis.set_extent((lon_ini, lon_fin, lat_ini, lat_fin), 
                                crs=data_projection)
            
            if extent_known:
                params['extent'] = (lon_ini, lon_fin, lat_ini, lat_fin)

                if wanted_projection is not None:
                    # solo para crs.PlateCarree(). Se necesita cartopy 0.18
                    axis.set_xticks(np.linspace(lon_ini, lon_fin, 10), 
                                    crs=wanted_projection)
                    axis.set_yticks(np.linspace(lat_ini, lat_fin, 10), 
                                    crs=wanted_projection)
                    params['transform'] = data_projection
                else:
                    axis.set_xticks(np.linspace(lon_ini, lon_fin, 10))
                    axis.set_yticks(np.linspace(lat_ini, lat_fin, 10))
                     
                if j == 0:
                    axis.set_ylabel("$\it{lat}$", fontsize=10)
                if i == rows - 1:
                    axis.set_xlabel("$\it{lon}$", fontsize=10)
                axis.tick_params(labelsize=8)

            if global_extent:
                axis.set_global()  

            im = axis.imshow(image, origin='lower', interpolation='nearest', 
                             cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, 
                             **params)

            if show_colorbar:
                divider = make_axes_locatable(axis)
                # the width of cax is 2% of axis and the padding is 0.1 inch
                cax = divider.append_axes("right", size="2%", pad=0.1, 
                                          axes_class=Axes)
                cb = fig.colorbar(im, ax=axis, cax=cax, drawedges=False, 
                                  format='%.0e')
                cb.outline.set_linewidth(0.1)
                cb.ax.tick_params(labelsize=8)
                if use_xarray:
                    cb.set_label(f'{data.name}[{data.units}]', rotation=90, 
                                 labelpad=10)

            if not show_axis:
                axis.set_axis_off()

    if save is not None and isinstance(save, str):
        savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0,
                transparent=transparent)
        close()
    else:
        show()


