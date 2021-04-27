import numpy as np
from numpy.lib.arraysetops import isin
import holoviews as hv
import xarray as xr
import hvplot.xarray 
import cartopy.crs as crs
import holoviews as hv
from matplotlib import cm, interactive

from .io import load_transform_mfdataset
from .utils import check_coords, slice_dataset
from .mpl_helpfunc import plot_mosaic_2d, plot_mosaic_3or4d

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

interactive_session = False

__all__ = ['plot_dataset',
           'plot_ndarray',
           'cm', 
           'crs',
           'set_interactive']


def _bold(string):
   return '\033[1m' + string + '\033[0m'


def set_interactive(state=True):
    global interactive_session
    interactive_session = state


def plot_ndarray(
    data, 
    interactive=None, 
    multichannel4d=False,
    show_colorbar=True, 
    share_colorbar=False,
    show_axis=True, 
    cmap='viridis', 
    share_dynamic_range=False,
    vmin=None, 
    vmax=None, 
    dpi=100,
    plot_size_px=500,
    dynamic=True,
    coastline=False,
    horizontal_padding=0.2,
    vertical_padding=0.1,
    max_static_subplot_rows=10,
    max_static_subplot_cols=10,
    subplot_titles=None,
    save=None,
    verbose=False, 
    ):
    """
    Plot a 2D, 3D or 4D ``numpy`` array or a tuple of 2D ``numpy`` arrays. 
    
    Parameters
    ----------
    data : numpy ndarray or tuple 
        2D, 3D or 4D ``numpy`` ndarray or a tuple of 2D ``numpy`` ndarrays. 
    interactive : bool, optional
        Whether to display an interactive (with ``bokeh``) or static (with
        ``matplotlib``) plot. In the case of a 3D ndarray, a slider will be used 
        to explore the data across time and/or vertical levels.
    multichannel4d : bool, optional
        If True, the dimensions of a 4D array are assumed [time, y, x, channels]
        which is useful for visualizing sequences of images with multiple 
        channels or variables. By default the order of the dimensions expected 
        in a 4D array are [time, z, y, x] to match the dimensions of ndarrays 
        contained in 4D xr.Datasets [time, level, lat, lon]. 
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
        interactive = True if interactive_session else False

    if isinstance(data, np.ndarray):
        data = np.squeeze(data)  # removing axes/dims of lenght one

    params1 = dict()
    if interactive:
        hv.extension('bokeh') # matplotlib is another option
        if isinstance(data, tuple):
            msg = '`data` is a tuple. This is supported only when `interactive=False`'
            raise ValueError(msg)
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            # Dataset((X, Y), Data)
            # X and Y are 1D arrays of shape M and N
            # Data is a ND array of shape NxM
            ds = hv.Dataset((range(data.shape[1]), range(data.shape[0]), data), 
                             ['x', 'y'], 'values')
            max_frames = 1
            sizexy_ratio = data.shape[1] / data.shape[0]
        elif isinstance(data, np.ndarray) and data.ndim == 3:
            params1['dynamic'] = dynamic
            # Dataset((X, Y, TIME), Data)
            # X and Y are 1D arrays of shape M and N
            # TIME is a 1D array of shape O
            # Data is a ND array of shape NxMxO
            ds = hv.Dataset((range(data.shape[2]), range(data.shape[1]),
                             range(data.shape[0]), data), 
                             ['x', 'y', 'time'], 'values')
            max_frames = data.shape[0]
            sizexy_ratio = data.shape[2] / data.shape[1]
        elif isinstance(data, np.ndarray) and data.ndim == 4:
            params1['dynamic'] = dynamic
            if multichannel4d:
                # adding a channel dimension
                ds = hv.Dataset((range(data.shape[3]), range(data.shape[2]),
                                 range(data.shape[1]), range(data.shape[0]), data),
                                ['channels', 'x', 'y', 'time'], 'values')
                max_frames = data.shape[0] * data.shape[3]
                sizexy_ratio = data.shape[2] / data.shape[1]
            else:
                # adding a level dimension
                ds = hv.Dataset((range(data.shape[3]), range(data.shape[2]),
                                 range(data.shape[1]), range(data.shape[0]), data),
                                ['x', 'y', 'level', 'time'], 'values')
                max_frames = data.shape[0] * data.shape[1]
                sizexy_ratio = data.shape[3] / data.shape[2]
        else:
            raise TypeError('`data` must be 3D or 4D when interactive=True')

        if vmin == 'min':
            vmin = data.min()
        if vmax == 'max':
            vmax = data.max()
        
        params2 = dict()
        # not needed in recent version of holoviews (can take clim=None)
        if vmin is not None and vmax is not None:
            params2['clim'] = (vmin, vmax)

        image_stack = ds.to(hv.Image, kdims=['x', 'y'], **params1)
        hv.output(backend='bokeh', dpi=dpi, max_frames=max_frames, widget_location='top')
        hv_cm = cmap if isinstance(cmap, str) else cmap.name        
        width = plot_size_px
        height = int(np.round(width / sizexy_ratio))

        # Compensating the width to accommodate the colorbar
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

        return image_stack.opts(hv.opts.Image(cmap=hv_cm,
                                              colorbar=show_colorbar,
                                              width=width, 
                                              height=height,
                                              tools=['hover'],
                                              **params2)) 

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
                    if not np.squeeze(data[i]).ndim == 2: # and data[i].shape[2] != 3: (excepting the case of 3 channels)
                        raise TypeError('tuple has non-2D arrays')

            return plot_mosaic_2d(
                data,
                show_colorbar=show_colorbar, 
                share_colorbar=share_colorbar,
                share_dynamic_range=share_dynamic_range,
                dpi=dpi, 
                plot_size_px=plot_size_px,
                cmap=cmap, 
                show_axis=show_axis, 
                save=save, 
                vmin=vmin, 
                vmax=vmax, 
                transparent=False, 
                coastline=coastline, 
                horizontal_padding=horizontal_padding,
                subplot_titles=subplot_titles,
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
            # max static subplots, assuming [time, level, lat, lon]
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
                dpi=dpi, 
                plot_size_px=plot_size_px,
                cmap=cmap, 
                show_axis=show_axis, 
                save=save, 
                vmin=vmin, 
                vmax=vmax, 
                transparent=False, 
                coastline=coastline, 
                mosaic_orientation=mosaic_orientation,
                horizontal_padding=horizontal_padding,
                vertical_padding=vertical_padding,
                subplot_titles=subplot_titles,
                verbose=verbose)

        else:
            raise TypeError('`data` must be a 2D/3D/4D ndarray or a tuple of 2D'
                            ' ndarrays when interactive=False')           


def plot_dataset(
    data, 
    interactive=None, 
    variable=None, 
    slice_time=None, 
    slice_level=None, 
    slice_lat=None, 
    slice_lon=None, 
    show_colorbar=True, 
    cmap='viridis', 
    logz=False, 
    share_dynamic_range=False, 
    vmin=None, 
    vmax=None, 
    wanted_projection=None, 
    data_projection=crs.PlateCarree(),
    coastline=False, 
    global_extent=False, 
    extent=None,
    dynamic=True, 
    slider_controls=False,
    dpi=80, 
    max_static_subplot_rows=10,
    max_static_subplot_cols=10,
    plot_size_px=600, 
    horizontal_padding=0.05,
    vertical_padding=0.15,
    verbose=True):
    """
    Plot an n-dimensional dataset (in-memory or from a path). The dataset is 
    loaded through ``xarray`` and therefore supports formats such as NetCDF, 
    IRIS or GRIB.

    Parameters
    ----------
    data : xarray Dataset/Dataarray or str
        Xarray (in memory) object or string with the path to the corresponding 
        NetCDF/IRIS/GRIB file. Expected dimensions: 2D [lat, lon], 3D array 
        [time, lat, lon] or 4D array [time, level, lat, lon].
    interactive : bool optional
        Whether to display an interactive plot (using ``hvplot``) with a 
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
    show_colorbar : bool optional
        Whether to show a colorbar.
    cmap : str or matplotlib.cm, optional
        Colormap, eg. viridis" or ecv.cm.viridis.  
    share_dynamic_range : bool optional
        Whether to share the dynamic range among mosaic subplots.
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
    extent : tuple of 4 floats
        A tuple with four values in the format (lon_ini, lon_fin, lat_ini, 
        lat_fin). Used to zoom the map to a given bounding box. Valid for static 
        plots, when coastline is shown. 
    dpi : int
        [interactive=False] DPI of the mosaic figure.
    plot_size_px : int optional
        Value of the figure width in pixels, used to scale up or down the size 
        of the ``hvplot`` or ``matplotlib`` figure.

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
    if interactive is None:
        interactive = True if interactive_session else False

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
    if variable is None: # taking the first >=2D data variable
        for i in data.data_vars:
            if data.data_vars.__getitem__(i).ndim >= 2:
                variable = i
                break
        if verbose == 1:
            print(f'The argument `variable` has not been set. Choosing '
                    f'`{variable}`, the first >=2D variable in `data` \n')
    elif isinstance(variable, int):
        variable = list(data.keys())[variable]
    else: # otherwise it is the variable name as a string
        if not isinstance(variable, str):
            raise ValueError('`variable` must be None, int or str')

    ### Getting info
    shape = data.data_vars.__getitem__(variable).shape
    if len(shape) >= 3:
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
    
    if var_array.ndim == 4: 
        dimp = '4D'
    elif var_array.ndim == 3: 
        dimp = '3D'
    elif var_array.ndim == 2:
        dimp = '2D'
    else:
        raise TypeError('Variable is neither 2D, 3D nor 4D')
 
    if verbose in [1, 2]:
        shape_slice = var_array.shape
        if hasattr(var_array, 'long_name'):
            print(f'{_bold("Name")} {variable}, {var_array.long_name}')
        else:
            print(f'{_bold("Name")} {variable}')
        if hasattr(var_array, 'units'):
            print(f'{_bold("Units:")} {var_array.units}') 
        print(f'{_bold("Dimensionality:")} {dimp}') 
        print(f'{_bold("Shape:")} {shape}')
        print(f'{_bold("Shape (sliced array):")} {shape_slice}')
        if dimp == '3D' or dimp == '4D':
            # assuming the min temporal sampling unit is minutes
            tini_slice = np.datetime_as_string(var_array.time[0].values, unit='m')
            tfin_slice = np.datetime_as_string(var_array.time[-1].values, unit='m') 
            print(f'{_bold("Time interval:")} {tini} --> {tfin}')
            print(f'{_bold("Time interval (sliced array):")} {tini_slice} --> {tfin_slice}\n')
    if verbose == 2:
        print(data.coords)
        print(data.data_vars, '\n')
    
    sizey = var_array.lat.shape[0]
    sizex = var_array.lon.shape[0]
    sizexy_ratio = sizex / sizey

    ### interactive plotting with slider(s) using bokeh
    if interactive:
        hv.extension('bokeh')
        if coastline or wanted_projection is not None:
            width = plot_size_px
            height = int(np.round(width / sizexy_ratio))
        else:
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
            shared_axes=True, 
            legend=True, 
            logz=logz, 
            widget_location='top', 
            project=project, 
            projection=wanted_projection, 
            global_extent=global_extent, 
            coastline=coastline, 
            **params)
        
    ### Static mosaic with matplotlib
    else:                
        if share_dynamic_range:
            if vmin is None:
                vmin = var_array.min().compute()
                vmin = np.array(vmin)
            if vmax is None:
                vmax = var_array.max().compute()   
                vmax = np.array(vmax)
        
        return plot_mosaic_3or4d(
            var_array, 
            show_colorbar=show_colorbar, 
            dpi=dpi, 
            plot_size_px=plot_size_px,
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
            extent=extent,
            horizontal_padding=horizontal_padding,
            vertical_padding=vertical_padding,
            verbose=verbose)
                
