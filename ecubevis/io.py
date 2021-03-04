from glob import glob
import xarray as xr


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = ['load_transform_mfdataset']


def load_transform_mfdataset(dir_path, dim, transform_func=None, 
                             transform_params={}):
    """
    Read a multi-file distributed N-dimensional ``xarray`` dataset.

    Parameters
    ----------
    dir_path : str
        Path to the files. 
    dim : str
        Dimension to concatenate along.
    transform_func : function or None
        Transform function to be applied to each file before loading it. 
    transform_params : dict, optional
        Parameters of the transform function.

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

    dir_path = dir_path + '/*.nc'
    paths = sorted(glob(dir_path))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined



