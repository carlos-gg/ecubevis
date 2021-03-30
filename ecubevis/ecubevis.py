import numpy as np
from numpy.lib.arraysetops import isin
import holoviews as hv
import xarray as xr
import hvplot.xarray 
import cartopy.crs as crs
import holoviews as hv
import matplotlib.colors as colors
from matplotlib.pyplot import colorbar, show, savefig, close, subplots, Axes
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
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
    interactive=False, 
    multichannel4d=False,
    show_colorbar=True, 
    share_colorbar=False,
    show_axis=True, 
    cmap='viridis', 
    share_dynamic_range=False,
    vmin=None, 
    vmax=None, 
    dpi=80,
    plot_size_px=600,
    coastline=False,
    horizontal_padding=0.1,
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
        [interactive=True] Value of the figure width in pixels, used to scale up 
        or down the size of the ``holoviews`` figure.
    subplot_titles : tuple of str
        [interactive=False] When the input data is a tuple of 2D ndarrays, the
        subplot_titles corresponds to a tuple of the same lenght with the titles
        of each subplot. 

    Returns
    -------
    Holoviews object when interactive=True, or None when interactive=False. In
    both cases the plot is shown on Jupyterlab.
    """
    if interactive:
        hv.extension('bokeh') # matplotlib is another option
        if isinstance(data, tuple):
            msg = '`data` is a tuple. This is supported only when `interactive=False`'
            raise ValueError(msg)
        elif isinstance(data, np.ndarray) and data.ndim == 3:
            # Dataset((X, Y, Z), Data), where
            # X is a 1D array of shape M ,
            # Y is a 1D array of shape N and
            # Z is a 1D array of shape O
            # Data is a ND array of shape NxMxO
            ds = hv.Dataset((range(data.shape[2]), range(data.shape[1]),
                             range(data.shape[0]), data), 
                             ['x', 'y', 'time'], 'values')
            max_frames = data.shape[0]
            sizexy_ratio = data.shape[2] / data.shape[1]
        elif isinstance(data, np.ndarray) and data.ndim == 4:
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
        
        params = dict()
        # not needed in recent version of holoviews (can take clim=None)
        if vmin is not None and vmax is not None:
            params['clim'] = (vmin, vmax)

        image_stack = ds.to(hv.Image, kdims=['x', 'y'], dynamic=True)
        hv.output(backend='bokeh', dpi=dpi, max_frames=max_frames, widget_location='top')
        hv_cm = cmap if isinstance(cmap, str) else cmap.name        
        width = plot_size_px
        height = int(np.round(width / sizexy_ratio))

        # Compensating the width to accommodate the colorbar
        if colorbar:
            cb_wid = 15
            cb_pad = 5
            tick_len = len(str(int(data.max())))
            if tick_len < 4:
                cb_tick = 25
            elif tick_len == 4:
                cb_tick = 35
            elif tick_len > 4:
                cb_tick = 45
            width += cb_pad + cb_wid + cb_tick

        return image_stack.opts(hv.opts.Image(cmap=hv_cm,
                                              colorbar=show_colorbar,
                                              colorbar_opts={'width': 15,
                                                             'padding': 5},
                                              width=width, 
                                              height=height,
                                              tools=['hover'],
                                              **params)) 

    # Non-interactive (static) matplotlib plot
    else:
        if isinstance(data, np.ndarray):
            data = np.squeeze(data)  # removing axes/dims of lenght one

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

            return _plot_mosaic_2d(
                data,
                show_colorbar=show_colorbar, 
                share_colorbar=share_colorbar,
                share_dynamic_range=share_dynamic_range,
                dpi=dpi, 
                cmap=cmap, 
                show_axis=show_axis, 
                save=save, 
                vmin=vmin, 
                vmax=vmax, 
                transparent=False, 
                coastline=coastline, 
                horizontal_padding=horizontal_padding,
                subplot_titles=subplot_titles)
        
        # Plotting a 3D or 4D array
        elif isinstance(data, np.ndarray):
            # max static subplots, assuming [time, level, lat, lon]  
            if data.ndim == 3:
                if verbose:
                    print('Plotting a single 3D np.ndarray')
                if data.shape[0] > max_static_subplot_rows:
                    data = data[:max_static_subplot_rows]
                    mosaic_orientation = 'col'
            if data.ndim == 4:
                if verbose:
                    print('Plotting a single 4D np.ndarray')
                if data.shape[0] > max_static_subplot_rows:
                    data = data[:max_static_subplot_rows]
                if data.shape[1] > max_static_subplot_cols:
                    data = data[:, :max_static_subplot_cols]
                mosaic_orientation = 'col'

            if share_dynamic_range:
                if vmin is None:
                    vmin = data.min()
                    vmin = np.array(vmin)
                if vmax is None:
                    vmax = data.max() 
                    vmax = np.array(vmax)
            
            return _plot_mosaic_3or4d(
                data, 
                show_colorbar=show_colorbar, 
                dpi=dpi, 
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
                subplot_titles=subplot_titles)

        else:
            raise TypeError('`data` must be a 2D/3D/4D ndarray or a tuple of 2D'
                            ' ndarrays when interactive=False')           


def plot_dataset(
    data, 
    interactive=True, 
    variable=None, 
    slice_time=None, 
    slice_level=None, 
    slice_lat=None, 
    slice_lon=None, 
    show_colorbar=True, 
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
        [interactive=True] Value of the figure width in pixels, used to scale up 
        or down the size of the ``hvplot`` figure.

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
    if variable is None: # taking the first 2D, 3D or 4D data variable
        for i in data.data_vars:
            if data.data_vars.__getitem__(i).ndim >= 2:
                variable = i
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
        
        return _plot_mosaic_3or4d(
            var_array, 
            show_colorbar=show_colorbar, 
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
            extent=extent,
            horizontal_padding=horizontal_padding,
            vertical_padding=vertical_padding)
                

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
    mosaic_orientation='col',
    horizontal_padding=0.05,
    vertical_padding=0.05,
    subplot_titles=None):
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

    # use_xarray=True for plot_dataset function
    if use_xarray:
        sizexy_ratio = data.lon.shape[0] / data.lat.shape[0]
        if data.ndim == 4 and 'level' in data.coords:
            cols = data.level.shape[0]
        elif data.ndim == 3:
            cols = 1
            rows = data.time.shape[0]
        else:
            cols = 1
            rows = 1
    # use_xarray=False for plot_ndarray function
    else:
        if data.ndim == 3:
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
        elif data.ndim == 2:
            sizexy_ratio = data.shape[0] / data.shape[1]
            cols = 1
            rows = 1

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

    colorbarzone = 1.4 if show_colorbar else 1 
    if mosaic_orientation == 'row' and data.ndim == 3:
        figsize = (max(8, rows*2) * sizexy_ratio * colorbarzone, max(8, cols*2)) 
    else:
        figsize = (max(8, cols*2) * sizexy_ratio * colorbarzone, max(8, rows*2)) 
    
    if wanted_projection is None and extent_known:
        wanted_projection = data_projection
        print(f'Assuming {wanted_projection} projection')

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
                if use_xarray and data.ndim != 2:
                    time = np.datetime64(image.time.values, 'm')
                    axis.set_title(f'$\ittime$={time}', fontsize=10)
            elif rows == 1:
                axis = ax[j]
                image = data[j]
                if use_xarray:
                    level = image.level.values
                    axis.set_title(f'$\itlevel$={level}', fontsize=10)
                else:
                    if mosaic_orientation == 'row' and subplot_titles is not None:
                        axis.set_title(subplot_titles[j], fontsize=10)
            else:
                axis = ax[i, j]
                image = data[i, j]              
                if use_xarray:
                    time = np.datetime64(image.time.values, 'm')
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

            if coastline and isinstance(axis, GeoAxes):
                axis.coastlines()
                axis.set_extent((lon_ini, lon_fin, lat_ini, lat_fin), 
                                crs=data_projection)
            
            if extent_known:
                params['extent'] = (lon_ini, lon_fin, lat_ini, lat_fin)
                params['transform'] = data_projection
                if wanted_projection is not None and \
                   wanted_projection == crs.PlateCarree():
                    # Cartopy 0.18 needed for other projections
                    axis.set_xticks(np.linspace(lon_ini, lon_fin, 7), 
                                    crs=wanted_projection)
                    axis.set_yticks(np.linspace(lat_ini, lat_fin, 7), 
                                    crs=wanted_projection)
                    lonform = LongitudeFormatter(number_format='.1f', 
                                                 degree_symbol='º')
                    latform = LatitudeFormatter(number_format='.1f', 
                                                degree_symbol='º')
                    axis.xaxis.set_major_formatter(lonform)
                    axis.yaxis.set_major_formatter(latform)

                    if j == 0:
                        axis.set_ylabel("$\it{lat}$", fontsize=10)
                    if i == rows - 1:
                        axis.set_xlabel("$\it{lon}$", fontsize=10)
                     
            axis.tick_params(labelsize=6)

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
                                  format=None) #format='%1.2e'
                cb.outline.set_linewidth(0.1)
                cb.ax.tick_params(labelsize=8)
                if use_xarray:
                    cb.set_label(f'{data.name} [{data.units}]', rotation=90, 
                                 labelpad=10)

            if not show_axis:
                axis.set_axis_off()
            else:
                axis.spines["right"].set_linewidth(0.1)
                axis.spines["left"].set_linewidth(0.1)
                axis.spines["top"].set_linewidth(0.1)
                axis.spines["bottom"].set_linewidth(0.1)

    if show_colorbar:
        horizontal_padding += 0.02
    fig.subplots_adjust(wspace=horizontal_padding, hspace=vertical_padding)

    if save is not None and isinstance(save, str):
        savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0.2,
                transparent=transparent)
        close()
    else:
        show()


def _plot_mosaic_2d(
    data, 
    show_colorbar=True, 
    share_colorbar=False,
    share_dynamic_range=False,
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
    horizontal_padding=0.05,
    subplot_titles=None):
    """
    """
    params = dict()
    if isinstance(data, tuple):
        tuple_data = []
        for i in range(len(data)):
            tuple_data.append(np.squeeze(data[i]))
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        tuple_data = [data]
    else:
        raise TypeError('`data` must be a single 2D array or tuple of 2D arrays')
    
    if share_colorbar:
        if (vmin is None or vmax is None) and not share_dynamic_range:
            raise ValueError('When `share_colorbar=True`, `vmin` and `vmax` '
                             'must be given or `share_dynamic_range=True`')

    if share_dynamic_range:
        minvals = [im.min() for im in tuple_data]
        vmin = np.min(minvals)
        maxvals = [im.max() for im in tuple_data]
        vmax = np.max(maxvals)

    first2d = tuple_data[0]
    sizexy_ratio = first2d.shape[1] / first2d.shape[0]

    if extent is not None:
        lon_ini, lon_fin, lat_ini, lat_fin = extent
        extent_known = True
    else:
        extent_known = False

    colorbarzone = 1.4 if show_colorbar else 1 
    cols = len(tuple_data)
    figsize = (8 * sizexy_ratio * colorbarzone, max(8, cols*2)) 

    if wanted_projection is None and extent_known:
        wanted_projection = data_projection
        print(f'Assuming {wanted_projection} projection')

    fig, ax = subplots(1, cols, sharex='col', dpi=dpi, figsize=figsize, 
                       constrained_layout=False, 
                       subplot_kw={'projection': wanted_projection})

    for j in range(cols):
        image = tuple_data[j]
        if cols == 1:
            axis = ax
            if subplot_titles is not None:
                axis.set_title(subplot_titles, fontsize=10)
        else:
            axis = ax[j]
            if subplot_titles is not None:
                axis.set_title(subplot_titles[j], fontsize=10)
       
        if logscale:
            image += np.abs(image.min())
            if vmin is None:
                linthresh = 1e-2
            else:
                linthresh = vmin
                norm = colors.SymLogNorm(linthresh)   
        else:
            norm = None                

        if coastline and isinstance(axis, GeoAxes):
            axis.coastlines()
            axis.set_extent((lon_ini, lon_fin, lat_ini, lat_fin), 
                            crs=data_projection)
        
        if extent_known:
            params['extent'] = (lon_ini, lon_fin, lat_ini, lat_fin)
            params['transform'] = data_projection
            if wanted_projection is not None and \
                wanted_projection == crs.PlateCarree():
                # Cartopy 0.18 needed for other projections
                axis.set_xticks(np.linspace(lon_ini, lon_fin, 7), 
                                crs=wanted_projection)
                axis.set_yticks(np.linspace(lat_ini, lat_fin, 7), 
                                crs=wanted_projection)
                lonform = LongitudeFormatter(number_format='.1f', 
                                                degree_symbol='º')
                latform = LatitudeFormatter(number_format='.1f', 
                                            degree_symbol='º')
                axis.xaxis.set_major_formatter(lonform)
                axis.yaxis.set_major_formatter(latform)

                axis.set_ylabel("$\it{lat}$", fontsize=10)
                axis.set_xlabel("$\it{lon}$", fontsize=10)
                    
        axis.tick_params(labelsize=6)

        if global_extent:
            axis.set_global()  

        im = axis.imshow(image, origin='lower', interpolation='nearest', 
                         cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, **params)

        if show_colorbar: 
            if not share_colorbar:
                divider = make_axes_locatable(axis)
                # the width of cax is 2% of axis and the padding is 0.1 inch
                cax = divider.append_axes("right", size="2%", pad=0.1, 
                                          axes_class=Axes)
                cb = fig.colorbar(im, ax=axis, cax=cax, drawedges=False, 
                                  format=None) #format='%1.2e'
                cb.outline.set_linewidth(0.1)
                cb.ax.tick_params(labelsize=8)
            else:
                if j+1 == cols:
                    fig.subplots_adjust(right=0.8)
                    axpos = axis.get_position()
                    cbar_pad = axpos.width * cols * 0.01 
                    cbar_width = axpos.width * cols * 0.02
                    cax = fig.add_axes([axpos.x0 + axpos.width + cbar_pad, 
                                        axpos.y0, cbar_width, axpos.height])
                    cb = fig.colorbar(im, ax=axis, cax=cax, drawedges=False, 
                                      format=None) #format='%1.2e'
                    cb.outline.set_linewidth(0.1)
                    cb.ax.tick_params(labelsize=8)

        if not show_axis:
            axis.set_axis_off()
        else:
            axis.spines["right"].set_linewidth(0.1)
            axis.spines["left"].set_linewidth(0.1)
            axis.spines["top"].set_linewidth(0.1)
            axis.spines["bottom"].set_linewidth(0.1)

    if show_colorbar:
        horizontal_padding += 0.02 * len(data)
    fig.subplots_adjust(wspace=horizontal_padding)

    if save is not None and isinstance(save, str):
        savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0.2,
                transparent=transparent)
        close()
    else:
        show()


