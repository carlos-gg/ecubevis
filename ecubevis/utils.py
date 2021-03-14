import numpy as np

__all__ = ['slice_dataset',
           'fix_longitude',
           'fix_latitude']


def fix_longitude(data):
    """ 
    If the data central longitude is 180º (0º to 360º) then we set it to 0º 
    (-180º to 180º)
    """
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby(data.lon)
    return data


def fix_latitude(data):
    """
    Reverse along latitude, 90º to -90º -> -90º to 90º
    """
    return data.reindex(lat=data.lat[::-1])


def slice_dataset(data, slice_time=None, slice_level=None, slice_lat=None, 
                  slice_lon=None):
    """  
    Slice an N-dimensional ``xarray`` Dataset across its dimensions (time, level,
    lat or lon, if present). This function is able to wrap selection assuming 
    360 degrees, which is not strighformard with ``xarray``. 

    Parameters
    ----------
    data : xarray Dataset
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
    data : xarray Dataset
        When any of the arguments ``slice_time``, ``slice_level``, 
        ``slice_lat``, ``slice_lon`` is defined (not None) the returned 
        ``data`` is the sliced input. If the slicing arguments are None the
        function returns the input Dataset.

    """ 
    data = check_coords(data)

    if np.any(data.lon > 180):
        data = fix_longitude(data)

    if data.lat[0] > data.lat[-1]:
        data = fix_latitude(data)

    if slice_time is not None and 'time' in data.coords:
        if isinstance(slice_time, tuple) and isinstance(slice_time[0], str):
            data = data.sel(time=slice(*slice_time))
        elif isinstance(slice_time, tuple) and isinstance(slice_time[0], int):
            data = data.isel(time=slice(*slice_time))
    
    if slice_level is not None and 'level' in data.coords:
        data = data.isel(level=slice(*slice_level))

    if slice_lat is not None and 'lat' in data.coords:
        data = data.sel(lat=slice(*slice_lat))

    if slice_lon is not None and 'lon' in data.coords:
        data = data.sel(lon=slice(*slice_lon))

    return data


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

