# Welcome to the repository of ecubevis

`ecubevis`: Earth CUBE VISualization with Python. Intended for the interactive exploration of n-dimensional (2D, 3D or 4D spatio-temporal) arrays on Jupyterlab. Supports both ``xarray.Dataset/DataArray`` (with metadata) or ``numpy.ndarray`` objects. 

## For BSC-ES users

To use `ecubevis` on the POWER-CTE cluster (P9), use the following modules to solve the dependencies:

```
module use /gpfs/projects/bsc32/software/rhel/7.4/ppc64le/POWER9/modules/all/
module load Cartopy/0.17.0-foss-2018b-Python-3.7.0
module load jupyterlab/3.0.9-foss-2018b-Python-3.7.0
```

The modules should have the same names on Nord3 (just "use" the Nord3 modules repository). 

That's it, you are ready to import `ecubevis` from this folder (assumining you've clones or downloaded this git repository). 

## How to use

Import the library:

```python
import ecubevis as ecv
```

The two main functions in ``ecubevis`` are: 

* ``ecv.plot_ndarray``: For plotting an in-memory ``numpy.ndarray`` object with 2, 3 or 4 dimensions (ndarrays do not carry metadata so the dimensions are implicit). The dimensions expected are [lat, lon] for 2D arrays, [time, lat, lon] for 3D arrays or [time, level, lat, lon] for 4D arrays. Additionally, ``plot_ndarray`` can take a tuple of 2D ndarrays, even with different grid/image size.

* ``ecv.plot_dataset``: For plotting an in-memory ``xr.Dataset`` or ``xr.DataArray`` objects with 2, 3, or 4 dimensions. The dimensions expected are [lat, lon] for 2D arrays, [time, lat, lon] for 3D arrays or [time, level, lat, lon] for 4D arrays.  

## Screenshots

`ecubevis` will allow you to create:

* interactive plots of in-memory 3D and 4D ``xr.Dataset`` or ``xr.DataArray`` objects. The sliders, thanks to `hvplot`, allow easy exploration of the data as 2D arrays across the time and vertical level dimensions:

![1](./screenshots/ecubevis_1.png | width=300)

* static mosaics of in-memory 3D and 4D ``xr.Dataset`` or ``xr.DataArray`` objects:

![2](./screenshots/ecubevis_2.png | width=300)

* interactive plots of in-memory 3D and 4D ``numpy.ndarray`` objects. Composition is possible thanks to ``holowvies``:

![2](./screenshots/ecubevis_3.png | width=300)

* static plots of in-memory 2D, 3D and 4D ``numpy.ndarray`` objects:

![2](./screenshots/ecubevis_4.png | width=300)

* static plots of a tuple of in-memory 2D ``numpy.ndarray`` objects:

![2](./screenshots/ecubevis_5.png | width=300)