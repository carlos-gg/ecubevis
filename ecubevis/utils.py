import numpy as np

__all__ = ['slice_dataset']


def slice_dataset(ndarray, slice_time=None, slice_level=None, slice_lat=None, 
                  slice_lon=None):
    """  
    Slice an N-dimensional ``xarray`` Dataset across its dimensions (time, level,
    lat or lon, if present). This function is able to wrap selection assuming 
    360 degrees, which is not strighformard with ``xarray``. 

    Parameters
    ----------
    ndarray : xarray Dataset
        Input N-dimensional dataset.
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

    Returns
    -------
    ndarray : xarray Dataset
        When any of the arguments ``slice_time``, ``slice_level``, 
        ``slice_lat``, ``slice_lon`` is defined (not None) the returned 
        ``ndarray`` is the sliced input. If the slicing arguments are None the
        function returns the input Dataset.

    """ 
    ndarray = check_coords(ndarray)
    if slice_time is not None and 'time' in ndarray.coords:
        if isinstance(slice_time, tuple) and isinstance(slice_time[0], str):
            ndarray = ndarray.sel(time=slice(*slice_time))
        elif isinstance(slice_time, tuple) and isinstance(slice_time[0], int):
            ndarray = ndarray.isel(time=slice(*slice_time))
    
    if slice_level is not None and 'level' in ndarray.coords:
        ndarray = ndarray.isel(level=slice(*slice_level))
    
    if slice_lat is not None and 'lat' in ndarray.coords:
        ndarray = ndarray.sel(dict(lat=ndarray.lat[(ndarray.lat > slice_lat[0]) & 
                                                   (ndarray.lat < slice_lat[1])]))
    
    if slice_lon is not None and 'lon' in ndarray.coords:
        # -180 to 180
        if ndarray.lon[0].values < 0:
            ndarray = ndarray.sel(dict(lon=ndarray.lon[(ndarray.lon > slice_lon[0]) & 
                                                       (ndarray.lon < slice_lon[1])]))
        # 0 to 360
        else:
            ndarray = lon_wrap_slice(ndarray, wrap=360, indices=(slice_lon[0], 
                                                                 slice_lon[1]))
    
    return ndarray


def check_coords(ndarray):
    standard_names = ['lat', 'lon', 'level', 'time', 'lat', 'lon', 'lat', 'lon']
    alternative_names = ['latitude', 'longitude', 'height', 'frequency', 'y', 'x', 'Y', 'X']

    for c in ndarray.coords:
        if c not in standard_names + alternative_names:
            msg = f'Xarray Dataset/Dataarray contains unknown coordinates. '
            msg += f'Must be one of: {standard_names} or {alternative_names}'
            raise ValueError(msg)

    for i, altname in enumerate(alternative_names):
        if altname in ndarray.coords:
            ndarray = ndarray.rename({alternative_names[i]: standard_names[i]})
    return ndarray


def lon_wrap_slice(ds, wrap=360, dim='lon', indices=(-25,45)):
    """
    wrap selection assuming 360 degrees. xarray core is not likely to include 
    this. Shame as would make nice method. Maybe someone builds something on 
    xarray that supports geographical co-ordinates..
    
    Parameters
    ----------
    :param ds: dataset
    :param wrap: default 360 -- period of co-ordinate.
    
    Notes
    -----
    https://github.com/pangeo-data/pangeo/issues/670
    https://gis.stackexchange.com/questions/205871/xarray-slicing-across-the-antimeridian

    # adjust the coordinate system of your data so it is centered over the anti-meridian instead
    # https://gis.stackexchange.com/questions/205871/xarray-slicing-across-the-antimeridian
    # t2m_83 = t2m_83.assign_coords(lon=(t2m_83.lon % 360)).roll(lon=(t2m_83.dims['lon'] // 2))

    """    
    sliceObj = slice(*indices)

    l = (ds[dim] + 2 * wrap) % wrap
    lstart = (sliceObj.start + 2 * wrap) % wrap
    lstop = (sliceObj.stop + 2 * wrap) % wrap
    if lstop > lstart:
        ind = (l <= lstop) & (l >= lstart)
    else:
        ind = (l <= lstop) | (l >= lstart)

    result = ds.isel(**{dim: ind})

    # need to make data contigious which may require rolling it and adjusting 
    # co-ord values.

    # step 1 work out how much to roll co-ordinate system. Need to find "break"
    # break is last place in array where ind goes from False to True
    brk = np.max(np.where(ind.values[:-1] != ind.values[1:]))
    brk = len(l) - brk - 1
    result = result.roll(**{dim: brk})
    # step 2 fix co-ords
    v = result[dim].values
    v = np.where(v < 180., v, v - 360)
    result = result.assign_coords(**{dim: v})

    return result
