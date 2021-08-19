import numpy as np

__all__ = ['slice_dataset',
           'fix_longitude',
           'fix_latitude']


COORDS_STAND = ['lat', 'lon', 'level', 'time', 'lat', 'lon', 'lat', 'lon']
COORDS_ALT = ['latitude', 'longitude', 'height', 'frequency', 'y', 'x', 'Y', 'X']


def fix_longitude(data, dim_name='lon'):
    """ 
    If the data central longitude is 180º (0º to 360º) then we set it to 0º 
    (-180º to 180º)
    """
    data.coords[dim_name] = (data.coords[dim_name] + 180) % 360 - 180
    data = data.sortby(data.lon)
    return data


def fix_latitude(data, dim_name='lat'):
    """
    Reverse along latitude, 90º to -90º -> -90º to 90º
    """
    if dim_name == 'lat':
        return data.reindex(lat=data.coords[dim_name][::-1])
    elif dim_name == 'latitude':
        return data.reindex(latitude=data.coords[dim_name][::-1])


def slice_dataset(data, slice_time=None, slice_level=None, slice_lat=None, 
                  slice_lon=None, drop_dates=False):
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
    drop_dates : bool, optional
        If True the time interval in ``slice_time`` will be removed. The default
        is False, meaning that the time interval is selected.

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
            if drop_dates:
                data = data.sel(time=~((data.time.astype(str) >= slice_time[0]) & (data.time.astype(str) <= slice_time[1])))
            else:
                data = data.sel(time=slice(*slice_time))
        elif isinstance(slice_time, tuple) and isinstance(slice_time[0], int):
            if drop_dates:
                data = data.drop_isel(time=list(range(slice_time[0], slice_time[1])))
            else:
                data = data.isel(time=slice(*slice_time))
    
    if slice_level is not None and 'level' in data.coords:
        data = data.isel(level=slice(*slice_level))

    if slice_lat is not None and 'lat' in data.coords:
        data = data.sel(lat=slice(*slice_lat))

    if slice_lon is not None and 'lon' in data.coords:
        data = data.sel(lon=slice(*slice_lon))

    return data


def check_coords(dataset, allow_unknown_coords=True):
    """
    """
    if not allow_unknown_coords:
        for c in dataset.coords:
            if c not in COORDS_STAND + COORDS_ALT:
                msg = f'Xarray Dataset/Dataarray contains unknown coordinates. '
                msg += f'Must be one of: {COORDS_STAND} or {COORDS_ALT}'
                raise ValueError(msg)

    for i, altname in enumerate(COORDS_ALT):
        if altname in dataset.coords:
            dataset = dataset.rename({COORDS_ALT[i]: COORDS_STAND[i]})
    return dataset

