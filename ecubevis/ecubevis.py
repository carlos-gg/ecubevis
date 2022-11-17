import numpy as np
import holoviews as hv
import xarray as xr
import hvplot.xarray 
import cartopy.crs as crs
import holoviews as hv
from matplotlib import cm, interactive
import matplotlib.pyplot as plt

from .utils import check_coords, slice_dataset
from .mpl_helpfunc import plot_mosaic_2d, plot_mosaic_3or4d

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

INTERACTIVE_SESSION = True 
DIMS2D = ('y', 'x')
DIMS3D = ('time', 'y', 'x')
DIMS4D = ('time', 'dim', 'y', 'x')
DIMS5D = ('time', 'dim', 'y', 'x', 'channel')

__all__ = ['plot',
           'plot_dataset',
           'plot_ndarray',
           'cm', 
           'crs',
           'set_interactive']


def _bold(string):
    return '\033[1m' + string + '\033[0m'


def set_interactive(state=True):
    global INTERACTIVE_SESSION
    INTERACTIVE_SESSION = state


def _get_maxframes(data, dimensions):
    maxframes = 1
    for i, dim in enumerate(dimensions):
        if dim not in ['lat', 'lon']:
            maxframes *= data.shape[i]
    return maxframes

def _get_xy_ratio(data, dimensions):
    for i, dim in enumerate(dimensions):
        if dim == 'y':
            y = i
        elif dim == 'x':
            x = i
    return data.shape[x] / data.shape[y]
    

def plot(data, variable=None, **kwargs):
    """
    Plot a 2D, 3D, 4D or 5D ``numpy`` array or a tuple of 2D ``numpy`` arrays, 
    or a 2D, 3D or 4D ``xarray`` Dataset/DataArray.

    Parameters
    ----------
    data : tuple or numpy.ndarray or xarray.Dataset or xr.DataArray 
        Input data.
    **kwargs : dict
        Arguments passed to the ``plot_ndarray`` or ``plot_dataset`` functions.
    """
    if isinstance(data, tuple):
        if 'interactive' not in kwargs.keys():
            kwargs['interactive'] = False
        if isinstance(data[0], np.ndarray):
            out = plot_ndarray(data, **kwargs)
        elif isinstance(data[0], (xr.Dataset, xr.DataArray)):
            out = plot_dataset(data, variable=variable, **kwargs)
    elif isinstance(data, np.ndarray):
        out = plot_ndarray(data, **kwargs)
    elif isinstance(data, (xr.Dataset, xr.DataArray)):
        out = plot_dataset(data, variable=variable, **kwargs)
    else:
        msg = f'`data` type (received{type(data)}) does not match any of the '
        msg += 'accepted types: tuple, numpy.ndarray or xarray.Dataset/DataArray'
        raise TypeError(msg)
    return out


def plot_ndarray(
    data, 
    interactive=None, 
    dimensions='auto',
    show_colorbar=True, 
    share_colorbar=False,
    show_axis=True, 
    cmap='RdBu_r', 
    share_dynamic_range=False,
    vmin=None, 
    vmax=None, 
    norm=None,
    dpi=100,
    plot_size_px=500,
    dynamic=True,
    show_coastline=False,
    horizontal_padding=0.2,
    vertical_padding=0.1,
    max_static_subplot_rows=10,
    max_static_subplot_cols=10,
    subplot_titles=None,
    plot_title=None,
    save=None,
    verbose=False, 
    ):
    """
    Plot a 2D, 3D, 4D or 5D ``numpy`` array or a tuple of 2D ``numpy`` arrays. 
    
    Parameters
    ----------
    data : numpy ndarray or tuple 
        2D, 3D, 4D or 5D ``numpy`` ndarray or a tuple of 2D ``numpy`` ndarrays. 
    interactive : bool, optional
        Whether to display an interactive (with ``bokeh``) or static (with
        ``matplotlib``) plot. In the case of a 3D ndarray, a slider will be used 
        to explore the data across time and/or additional dimensions.
    dimensions : tuple or str, optional
        The dimensions of the numpy array. Used when ``interactive`` is True. If
        'auto' then the global variables DIMS2D, DIMS3D, DIMS4D or DIMS5D are 
        used.
    colorbar : bool optional
        Whether to show the colorbar.
    show_axis : bool optional
        Whether to show the axis.
    cmap : str or matplotlib.cm, optional
        Colormap, eg. viridis" or ecv.cm.viridis. 
    share_dynamic_range : bool optional
        Whether to share the dynamic range among mosaic subplots, taking the 
        min and max values of all the subplots. Overrides the values of ``vmin`` 
        and ``vmax``.
    vmin : float or None
        Min value to be displayed.
    vmax : float or None
        Max value to be displayed.
    dpi : int optional
        [interactive=False] DPI of the mosaic figure.
    plot_size_px : int optional
        If ``interactive=True``, it sets the value of the figure width in pixels, 
        used to scale up or down the size of the ``holoviews`` figure. If
        ``interactive=False``, then the width of the plot will be equal to 
        ``plot_size_px * number_of_columns`` and the height will be adjusted
        according to the image x/y ratio.
    subplot_titles : tuple of str
        [interactive=False] When the input data is a tuple of 2D ndarrays, the
        subplot_titles corresponds to a tuple of the same lenght with the titles
        of each subplot. 
    save : str
        [interactive=False] Path for saving the ``matplotlib`` figure to disk.
    verbose : bool optional
        Whether to display additional information. False by default.

    Returns
    -------
    Holoviews object when interactive=True, or None when interactive=False. In
    both cases the plot is shown on Jupyterlab.
    """
    if interactive is None:
        interactive = True if INTERACTIVE_SESSION else False

    if isinstance(data, np.ndarray):
        data = np.squeeze(data)  # removing axes/dims of lenght one

    params1 = dict()
    if interactive:
        hv.extension('bokeh')  # matplotlib could be used instead
        if isinstance(data, tuple):
            msg = '`data` is a tuple. This is supported only when `interactive=False`'
            raise ValueError(msg)
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                if dimensions == 'auto':
                    dimensions = list(DIMS2D)
                    if verbose:
                        print(f'`dimensions` is None, assuming `data` has {dimensions}')
                else:
                    dimensions = list(dimensions)
                sizexy_ratio = _get_xy_ratio(data, dimensions)
                max_frames = 1
                # Dataset((X, Y), Data)
                # X and Y are 1D arrays of shape M and N
                # Data is a ND array of shape NxM
                ds = hv.Dataset((range(data.shape[1]), range(data.shape[0]), data), 
                                dimensions[::-1] , 'values')
            elif data.ndim == 3:
                if dimensions == 'auto':
                    dimensions = list(DIMS3D)
                    if verbose:
                        print(f'`dimensions` is None, assuming `data` has {dimensions}')
                else:
                    dimensions = list(dimensions)
                max_frames = _get_maxframes(data, dimensions)
                sizexy_ratio = _get_xy_ratio(data, dimensions)
                params1['dynamic'] = dynamic
                ds = hv.Dataset((range(data.shape[2]), range(data.shape[1]), range(data.shape[0]), 
                                 data), dimensions[::-1] , 'values')
            elif data.ndim == 4:
                if dimensions == 'auto':
                    dimensions = list(DIMS4D)
                    if verbose:
                        print(f'`dimensions` is None, assuming `data` has {dimensions}')
                else:
                    dimensions = list(dimensions)
                max_frames = _get_maxframes(data, dimensions)
                sizexy_ratio = _get_xy_ratio(data, dimensions)
                params1['dynamic'] = dynamic
                ds = hv.Dataset((range(data.shape[3]), range(data.shape[2]), range(data.shape[1]), 
                                 range(data.shape[0]), data),
                                dimensions[::-1] , 'values')
            elif data.ndim == 5:
                if dimensions == 'auto':
                    dimensions = list(DIMS5D)
                    if verbose:
                        print(f'`dimensions` is None, assuming `data` has {dimensions}')
                else:
                    dimensions = list(dimensions)
                max_frames = _get_maxframes(data, dimensions)
                sizexy_ratio = _get_xy_ratio(data, dimensions)
                params1['dynamic'] = dynamic
                ds = hv.Dataset((range(data.shape[4]), range(data.shape[3]), range(data.shape[2]), 
                                 range(data.shape[1]), range(data.shape[0]), data),
                                dimensions[::-1] , 'values')
        else:
            raise TypeError('`data` must be 2D/3D/4D/5D ndarray when interactive=True')

        if vmin == 'min':
            vmin = data.min()
        if vmax == 'max':
            vmax = data.max()
        
        image_stack = ds.to(hv.Image, kdims=['x', 'y'], **params1)
        hv.output(backend='bokeh', dpi=dpi, max_frames=max_frames, widget_location='top')
        hv_cm = cmap if isinstance(cmap, str) else cmap.name        
        width = plot_size_px
        height = int(np.round(width / sizexy_ratio))

        # Compensating the width to accommodate the colorbar
        params2 = dict()
        if show_colorbar:
            cb_wid = 15
            cb_pad = 5
            tick_len = len(str(data.max()))
            if tick_len < 4:
                cb_tick = 25
            elif tick_len == 4:
                cb_tick = 35
            elif tick_len > 4:
                cb_tick = 45
            width += cb_pad + cb_wid + cb_tick
            params2['colorbar_opts'] = {'width': cb_wid, 'padding': cb_pad}

        imstack = image_stack.opts(hv.opts.Image(cmap=hv_cm,
                                              colorbar=show_colorbar,
                                              width=width, 
                                              height=height,
                                              tools=['hover'],
                                              clim=(vmin, vmax), 
                                              **params2)) 
        # hv.save(imstack, 'test.gif', fmt='gif')  # creates html with animation 
        return imstack

    # Non-interactive (static) matplotlib plot
    else:
        # Plotting a single 2D ndarray or a tuple of 2D arrays
        if (isinstance(data, np.ndarray) and data.ndim == 2) or isinstance(data, tuple):
            if isinstance(data, np.ndarray) and data.ndim == 2:
                if verbose:
                    print('Plotting a single 2D np.ndarray')

            elif isinstance(data, tuple):
                if verbose:
                    print('Plotting a tuple of 2D np.ndarrays')
                for i in range(len(data)):
                    # checking the elements are 2d 
                    if not np.squeeze(data[i]).ndim == 2: 
                        raise TypeError('tuple has non-2D arrays')

            return plot_mosaic_2d(
                data,
                show_colorbar=show_colorbar, 
                share_colorbar=share_colorbar,
                share_dynamic_range=share_dynamic_range,
                dpi=dpi, 
                plot_size_px=plot_size_px,
                cmap=cmap, 
                norm=norm,
                show_axis=show_axis, 
                save=save, 
                vmin=vmin, 
                vmax=vmax, 
                transparent=False, 
                coastline=show_coastline, 
                horizontal_padding=horizontal_padding,
                subplot_titles=subplot_titles,
                plot_title=plot_title,
                verbose=verbose)
        
        # Plotting a 3D or 4D array
        elif isinstance(data, np.ndarray):  
            # max static subplots, assuming [time, lat, lon]
            if data.ndim == 3:
                if verbose:
                    print('Plotting a single 3D np.ndarray')
                if data.shape[0] > max_static_subplot_rows:
                    data = data[:max_static_subplot_rows]
                mosaic_orientation = 'row'
            # max static subplots, assuming [time, 4th_dim, lat, lon]
            elif data.ndim == 4:
                if verbose:
                    print('Plotting a single 4D np.ndarray')
                if data.shape[0] > max_static_subplot_rows:
                    data = data[:max_static_subplot_rows]
                if data.shape[1] > max_static_subplot_cols:
                    data = data[:, :max_static_subplot_cols]
                mosaic_orientation = 'col'
            else:
                raise TypeError('`data` must be a 2D/3D/4D ndarray or a tuple '
                            ' of 2D ndarrays when interactive=False')

            if share_dynamic_range:
                if vmin is None:
                    vmin = data.min()
                    vmin = np.array(vmin)
                if vmax is None:
                    vmax = data.max() 
                    vmax = np.array(vmax)
            
            return plot_mosaic_3or4d(
                data, 
                show_colorbar=show_colorbar, 
                share_colorbar=share_colorbar,
                share_dynamic_range=share_dynamic_range,
                dpi=dpi, 
                plot_size_px=plot_size_px,
                cmap=cmap, 
                norm=norm,
                show_axis=show_axis, 
                save=save, 
                vmin=vmin, 
                vmax=vmax, 
                transparent=False, 
                coastline=show_coastline, 
                mosaic_orientation=mosaic_orientation,
                horizontal_padding=horizontal_padding,
                vertical_padding=vertical_padding,
                subplot_titles=subplot_titles,
                plot_title=plot_title,
                verbose=verbose)

        else:
            raise TypeError('`data` must be a 2D/3D/4D ndarray or a tuple of 2D'
                            ' ndarrays when interactive=False')           
        

def plot_dataset(
    data, 
    variable=None, 
    interactive=None, 
    slice_time=None, 
    slice_level=None, 
    slice_lat=None, 
    slice_lon=None, 
    show_colorbar=True, 
    cmap='RdBu_r', 
    show_grid=True,
    show_axis=True,
    norm=None,
    vmin=None, 
    vmax=None, 
    wanted_projection=None, 
    data_projection=None,
    show_coastline=False, 
    global_extent=False, 
    extent=None,
    col=None,
    row=None,
    col_wrap=None,
    dynamic=True, 
    slider_controls=False,
    dpi=100, 
    plot_size_px=400, 
    colorbar_pad=0.05,
    colorbar_shrink=0.8,
    colorbar_orientation='v',
    colorbar_aspect=30,
    subplot_titles=None,
    ylabel_pad=-0.16,
    xlabel_pad=-0.16,
    verbose=True):
    """
    Plot a 2D, 3D or 4D ``xarray`` Dataset/DataArray (e.g., NetCDF, IRIS or 
    GRIB file loaded in-memory). 

    Parameters
    ----------
    data : xarray.Dataset or xarray.Dataarray 
        In-memory Xarray object corresponding to a NetCDF/IRIS/GRIB file on disk. 
        Expected dimensions: 2D [lat, lon], 3D array [time, lat, lon] or 4D 
        array e.g. [time, level, lat, lon].
    interactive : bool optional
        Whether to display an interactive plot (using ``hvplot``) with a 
        slider or an static mosaic (using ``matplotlib``). 
    variable : str or int or None, optional
        This applies only to a data input with type xarray.Dataset. It 
        corresponds to the name of the variable to be plotted or the index at 
        which it is located. 
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
    show_colorbar : bool optional
        Whether to show a colorbar.
    cmap : str or matplotlib.cm, optional
        Colormap, eg. viridis" or ecv.cm.viridis.  
    vmin : float or None
        Min value to be displayed.
    vmax : float or None
        Max value to be displayed.
    wanted_projection : cartopy.crs projection, optional
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
    data_projection : cartopy.crs projection, optional
        E.g. crs.PlateCarree().
    extent : tuple of 4 floats
        A tuple with four values in the format (lon_ini, lon_fin, lat_ini, 
        lat_fin). Used to zoom the map to a given bounding box. Valid for static 
        plots, when coastline is shown. 
    col : str, optional
        Dimension to use on columns when interactive=False.
    row : str, optional
        Dimension to use on rows when interactive=False.
    col_wrap : int, optional
        Number of columns. 
    dpi : int
        [interactive=False] DPI of the mosaic figure.
    plot_size_px : int optional
        Value of the figure width in pixels, used to scale up or down the size 
        of the ``hvplot`` or ``matplotlib`` figure.
    slider_controls : bool, optional
        Whether to show the slider controls when ``interactive=True``.
    ylabel_pad : float, optional
        Padding for the location of the Y label (lat).
    xlabel_pad : float, optional
        Padding for the location of the X label (lon).

    Notes
    -----
    https://github.com/pydata/xarray/issues/2199
    https://hvplot.holoviz.org/user_guide/Gridded_Data.html
    https://hvplot.holoviz.org/user_guide/Geographic_Data.html    

    """     
    if interactive is None:
        interactive = True if INTERACTIVE_SESSION else False
    
    if not isinstance(data, (tuple, xr.Dataset, xr.DataArray)):
        raise TypeError('`data` must be a tuple of 2d grids or a single xr.Dataset/DataArray')  
    
    if isinstance(data, tuple):
        if np.squeeze(data[0].values).ndim > 2:
            raise ValueError(f'Plotting of a tuple of xr.DataArrays is supported only for 2d grids, got {np.squeeze(data[0].values).ndim}')
        for arr in data:
            arr = check_coords(arr)
        var_array = xr.concat(data, dim='channel')  # assuming all 2d grids have the same size
        if 'time' in var_array.coords:
            var_array = var_array.drop_vars('time')
        if 'level' in var_array.coords:
            var_array = var_array.drop_vars('level')
        shape = var_array.shape
    elif isinstance(data, xr.DataArray):
        var_array = check_coords(data)
        shape = var_array.shape
    elif isinstance(data, xr.Dataset):
        ### Selecting the variable 
        if variable is None: 
            raise ValueError('The argument `variable` has not been set. Pass a DataArray or set `variable` to \n' 
                             f'one of the {data.data_vars}')
        elif isinstance(variable, int):
            variable = list(data.keys())[variable]
        else: # otherwise it is the variable name as a string
            if not isinstance(variable, str):
                raise ValueError('`variable` must be None, int or str')
        
        ### Getting info
        shape = data.data_vars.__getitem__(variable).shape
        var_array = check_coords(data)
        var_array = var_array.data_vars.__getitem__(variable)
    
    if show_coastline and (wanted_projection is None or data_projection is None):
        print('When `show_coastline=True`, both `data_projection` and `wanted_projection` must be provided. Assuming crs.PlateCarree() projection')
        wanted_projection = crs.PlateCarree()
        data_projection = crs.PlateCarree()

    ### Slicing
    var_array = slice_dataset(var_array, slice_time, slice_level, slice_lat, slice_lon)    
 
    if verbose in [1, 2]:
        if hasattr(var_array, 'name') and var_array.name is not None:
            if hasattr(var_array, 'long_name'):
                print(f'{_bold("Variable name:")} {var_array.name}, {var_array.long_name}')
            else:
                print(f'{_bold("Variable name:")} {var_array.name}')
        if hasattr(var_array, 'units'):
            print(f'{_bold("Units:")} {var_array.units}') 
        print(f'{_bold("Dimensionality:")} {var_array.ndim}D') 
        print(f'{_bold("Shape:")} {shape}')
        if slice_lat is not None or slice_lon is not None or slice_time is not None or slice_level is not None:
            print(f'{_bold("Shape (sliced array):")} {var_array.shape}')
    if verbose == 2:
        print(data.coords)
        print(data.data_vars, '\n')
    
    sizexy_ratio = var_array.lon.shape[0] / var_array.lat.shape[0]
        
    ### interactive plotting with slider(s) using bokeh
    if interactive:
        hv.extension('bokeh')
        width = plot_size_px
        height = int(np.round(width / sizexy_ratio))
        params = dict(height=height, width=width)
        project = False if wanted_projection is None else True

        if slider_controls:
            params['widget_type'] = 'scrubber'
    
        return var_array.hvplot(
            kind='image', 
            x='lon',
            y='lat', 
            dynamic=dynamic, 
            colorbar=show_colorbar, 
            cmap=cmap, 
            clim=(vmin,vmax),
            shared_axes=True, 
            legend=True, 
            logz=True if norm == 'log' else False, 
            widget_location='top', 
            project=project, 
            projection=wanted_projection, 
            global_extent=global_extent, 
            coastline=show_coastline,
            **params)
        
    ### Static mosaic with matplotlib
    else:
        var_array = np.squeeze(var_array)
        plot_size_inches = plot_size_px / dpi     
        width = plot_size_inches
        height = int(np.round(width / sizexy_ratio))
        
        if var_array.ndim in [3, 4]: 
            if var_array.shape[0] > 10 or (var_array.ndim == 4 and var_array.shape[1] > 10):
                raise ValueError('Check the shape of your array. Try slicing it!')

        plt.rcParams["figure.dpi"] = dpi

        if show_colorbar:
            if colorbar_orientation == 'h':
                colorbar_orientation = 'horizontal'
            elif colorbar_orientation == 'v':
                colorbar_orientation = 'vertical'

            cbar_kwargs={"orientation": colorbar_orientation, "shrink": colorbar_shrink, 
                         "aspect": colorbar_aspect, "pad": colorbar_pad}
            if isinstance(data, tuple):
                cbar_kwargs['label'] = ''
        else:
            cbar_kwargs = None

        if var_array.ndim == 2:
            params = dict(cmap=cmap, vmin=vmin,vmax=vmax, add_colorbar=show_colorbar,
                          cbar_kwargs=cbar_kwargs)
        else:
            if row is None:
                if 'time' in var_array.coords and col != 'time':
                    print('`row` is None. `row` set to `time`')
                    row = 'time'
            if col is None:
                if 'level' in var_array.coords and row != 'level':
                    print('`col` is None. `col` set to `level`')
                    col = 'level'
            if col is None and row is None:
                if 'channel' in var_array.dims:
                    col = 'channel'
                else:
                    raise ValueError('Provide `row` or `col`')

            params = dict(row=row, col=col, norm=norm, col_wrap=col_wrap, 
                          cmap=cmap, vmin=vmin, vmax=vmax, size=plot_size_inches, 
                          aspect=sizexy_ratio if wanted_projection is None else plot_size_inches,
                          add_colorbar=show_colorbar, cbar_kwargs=cbar_kwargs)
        if data_projection is not None:
            params['transform'] = data_projection
        if wanted_projection is not None:
            params['subplot_kws'] = {"projection": wanted_projection}
        
        fig = var_array.plot(**params)

        if show_colorbar:
            if hasattr(fig, 'cbar'):
                fig.cbar.outline.set_linewidth(0.2)
            elif hasattr(fig, 'colorbar'): 
                fig.colorbar.outline.set_linewidth(0.2)
            if isinstance(data, tuple):
                fig.cbar.set_label('')
        
        # setting the map's options on all axes
        if extent is None:
            extent = (var_array.lon[0].values, var_array.lon[-1].values, 
                      var_array.lat[0].values, var_array.lat[-1].values - 1)

        if var_array.ndim == 2:
            axes = [fig.axes]
        elif var_array.ndim > 2:
            axes = fig.axes.flat
        
        if subplot_titles is not None:
            if subplot_titles == '':
                subplot_titles = len(axes) * [subplot_titles]
            if len(subplot_titles) != len(axes):
                raise ValueError(f'`subplot_titles` must be equal to {len(axes)}, got {len(subplot_titles)}')

        for i, ax in enumerate(axes):
            if subplot_titles is not None:
                ax.set_title(subplot_titles[i])
            if wanted_projection is not None and data_projection is not None:
                if show_coastline:
                    ax.coastlines() 
                if extent is not None:
                    ax.set_extent(extent)
                if global_extent:
                    ax.set_global()
                if show_axis: 
                    # not showing the ylabel and xlabel, why?
                    # <GeoAxesSubplot:title={'center':'time = 2013-01-01'}, xlabel='lon', ylabel='Latitude [degrees_north]'>
                    if col_wrap is not None and i % col_wrap == 0:
                        ax.text(ylabel_pad, 0.55, 'lat', rotation='vertical', transform=ax.transAxes)  
                    elif col_wrap is None:
                        ax.text(ylabel_pad, 0.55, 'lat', rotation='vertical', transform=ax.transAxes)                       
                    ax.text(0.5, xlabel_pad, 'lon', rotation='horizontal', transform=ax.transAxes)
                
                    # https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
                    gl = ax.gridlines(alpha=0.2, draw_labels=True, dms=True)
                    gl.ylines = show_grid
                    gl.xlines = show_grid
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlabel_style = dict(fontsize=7)
                    gl.ylabel_style = dict(fontsize=7)
            
            ax.spines["right"].set_linewidth(0.2)
            ax.spines["left"].set_linewidth(0.2)
            ax.spines["top"].set_linewidth(0.2)
            ax.spines["bottom"].set_linewidth(0.2)
            if "geo" in ax.spines:
                ax.spines["geo"].set_linewidth(0.2)

        return fig