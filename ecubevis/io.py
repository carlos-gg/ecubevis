from glob import glob
import xarray as xr
from joblib import Parallel, delayed


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = ['load_transform_mfdataset']


def load_transform_mfdataset(dir_path, dim='time', transform_func=None, 
                             transform_params={}, n_jobs=1):
    """
    Read a multi-file distributed N-dimensional ``xarray`` dataset.

    Parameters
    ----------
    dir_path : str
        Path to the files. 
    dim : str
        Dimension to concatenate along.
    transform_func : function or None
        Transform function to be applied to each xr.Dataset or xr.DataArray 
        before loading and concatenating it. 
    transform_params : dict, optional
        Parameters of the transform function.
    n_jobs : int, optional
        Number of jobs.

    Returns
    -------
    combined : xarray Dataset
        Combined dataset.

    Notes
    -----
    https://xarray.pydata.org/en/stable/io.html#reading-multi-file-datasets
    """
    def process_one_path(file_path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(file_path) as ds:
            if transform_func is not None:
                ds = transform_func(ds, **transform_params)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    if not dir_path.endswith('*.nc'):
        dir_path = dir_path + '/*.nc'
    paths = sorted(glob(dir_path))
    print(f'Pre-processing {len(paths)} paths with `{transform_func.__name__}` function using {n_jobs} jobs')
    datasets = Parallel(n_jobs=n_jobs)(delayed(process_one_path)(file_path) for file_path in paths)
    print(f'Concatenating the preprocessed datasets along {dim}')
    combined = xr.concat(datasets, dim)
    return combined



