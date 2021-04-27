from matplotlib.pyplot import show, savefig, close, subplots, Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.colors as colors
import cartopy.crs as crs
import numpy as np
import xarray as xr


def plot_mosaic_3or4d(
    data, 
    show_colorbar=True, 
    dpi=100, 
    plot_size_px=500,
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
    horizontal_padding=0.2,
    vertical_padding=0.05,
    subplot_titles=None,
    verbose=False):
    """
    
    Ticks with non-rectangular projection supported in Carotpy 0.18
    https://scitools.org.uk/cartopy/docs/latest/gallery/gridliner.html
    axis.gridlines(draw_labels=True)
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
            rows = data.time.shape[0]
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

    if show_colorbar:
        colorbarzone = 1.1
    else:
        colorbarzone = 1 
    plot_size_inches = plot_size_px / dpi 
    if mosaic_orientation == 'row' and data.ndim == 3:
        figsize = (plot_size_inches * cols, plot_size_inches / sizexy_ratio)  
    else:
        figsize = (plot_size_inches * cols * colorbarzone, plot_size_inches * rows / sizexy_ratio)
    
    if wanted_projection is None and extent_known:
        wanted_projection = data_projection
        if verbose:
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
                        if isinstance(subplot_titles, str):
                            axis.set_title(subplot_titles, fontsize=10)
                        elif isinstance(subplot_titles, tuple):
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
                if use_xarray and hasattr(data, 'units'):
                    cb.set_label(f'{data.name} [{data.units}]', rotation=90, 
                                 labelpad=10)

            if not show_axis:
                axis.set_axis_off()
            else:
                axis.spines["right"].set_linewidth(0.1)
                axis.spines["left"].set_linewidth(0.1)
                axis.spines["top"].set_linewidth(0.1)
                axis.spines["bottom"].set_linewidth(0.1)
                if "geo" in axis.spines:
                    axis.spines["geo"].set_linewidth(0.1)

    if show_colorbar:
        horizontal_padding += 0.02 * len(data)
    fig.subplots_adjust(wspace=horizontal_padding, hspace=vertical_padding)

    if save is not None and isinstance(save, str):
        savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0.2,
                transparent=transparent)
        close()
    else:
        show()


def plot_mosaic_2d(
    data, 
    show_colorbar=True, 
    share_colorbar=False,
    share_dynamic_range=False,
    dpi=100, 
    plot_size_px=600,
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
    horizontal_padding=0.2,
    subplot_titles=None,
    verbose=False):
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

    cols = len(tuple_data)
    plot_size_inches = plot_size_px / dpi 
    figsize = (plot_size_inches * cols, plot_size_inches / sizexy_ratio) 

    if wanted_projection is None and extent_known:
        wanted_projection = data_projection
        if verbose:
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


