# Welcome to the repository of ecubevis

`ecubevis`: Earth CUBE VISualization with Python. Intended for the interactive exploration of n-dimensional (2d, 3d or 4d spatio-temporal) arrays on Jupyterlab.

## For BSC-ES users

To use `ecubevis` on the POWER-CTE cluster (P9), use the following modules to solve the dependencies:

```
module use /gpfs/projects/bsc32/software/rhel/7.4/ppc64le/POWER9/modules/all/
module load Cartopy/0.17.0-foss-2018b-Python-3.7.0
module load jupyterlab/3.0.9-foss-2018b-Python-3.7.0
```

The modules should have the same names on Nord3 (just "use" the Nord3 modules repository). 

That's it, you are ready to import `ecubevis` from this folder (assumining you've clones or downloaded this git repository). 