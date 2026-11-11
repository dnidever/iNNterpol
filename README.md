# iNNterpol

This installable distribution is maintained by David Nidever from
[cwestend's original iNNterpol project](https://github.com/cwestend/iNNterpol).
The original authors developed the neural networks and supplied the trained
weights. This fork adds Python packaging and adjusts imports and checkpoint
loading for installation; the scientific interpolation calculations are preserved.
Please cite the original paper listed below when using this software.

### Neural Network Interpolator for Atlas9/MARCS/Phoenix stellar model atmospheres

![Temperature differences between original and obtained untrained (test) atmospheres](../assets/DTemp_chkCAE_16_71_M0_mp05_lgg45_scale.png?raw=true)
(Fig1. plot of Temperature differences in % between original and obtained for untrained (test) atmospheres of metal-rich dwarf stars.)

**In press!** The paper is accepted and in press in Astronomy & Astrophisics, if you use the code, please cite it by its DOI: https://doi.org/10.1051/0004-6361/202346372. A pre-print version is available at https://arxiv.org/abs/2306.06938.

NOTE: For MARCS model iNNterpolator visit:  https://github.com/cwestend/iNNterpol/tree/main/iNNterpol_MARCS
For PHOENIX model iNNterpolator visit:  https://github.com/cwestend/iNNterpol/tree/main/iNNterpol_PHOENIX

The aim is to obtain a fast and reliable way to recover the model atmospheres calculated for the ATLAS9 model data for the 
APOGEE sky survey (see http://research.iac.es/proyecto/ATLAS-APOGEE//). These atmospheres were calculated
for a 5-dimensional grid, dimensions being the  metalicity M/H, carbon abundance C/M, other-element abundance O/M, effective 
Temperature and (log) surface Gravity. The iNNterpol not only recovers the four atmospheric parameters of mass, temperature, 
pressure and electronic density at the grid points they were calculated for but also for intermediate values of these.

Using the ATLAS9 model data we trained a deep neural-network based on a previous Convolutional Auto-Encoder (CAE)
that reduces the features of these models to a reduced parameter space. 

A 16-layer deep NN, with 40 nodes on each layer showed to give optimal results as shown on the recovered errors
and the fluxes (line profiles) calculated from these recovered atmospheres.

(A similar approach with a 12 component Principal component analysis (PCA) obtained by Singular-Value Decomposition (SVD) analysis of these models
was also undertaken and is shown in https://github.com/cwestend/iNNterpol/tree/main/iNNterpol_PCA12).


The code is provided together with the trained weights of the CAE and NN in order to be able to rapidly reconstruct each model atmosphere.
The full training code is available also for full reproducibility: https://github.com/cwestend/iNNterpol/tree/main/training_MARCS 

It also provides an extremely fast way of interpolating models between the calculated values. Models for areas (or grid values) where no data
is available on one or more of the extremes of the grid values can also be recovered. These regions can be considered as 
unbounded and thus a linear interpolation cannot be applied. Our iNNterpol method is able to learn from the data and give 
a smooth and reasonable values on these regions provided a reasonable departure in distance from grid values that do have 
calculated model data. A study of these regions, together with the quality of the recovered models and the fluxes obtained 
from these shall be undertaken in a future paper.

## Installation

Requires Python 3.9 or newer. Install from PyPI:

```sh
python -m pip install innterpol
```

Alternatively, install directly from this fork:

```sh
python -m pip install "git+https://github.com/dnidever/iNNterpol.git"
```

To install a local checkout:

```sh
python -m pip install .
```

For development use `python -m pip install -e ".[test]"`.
NumPy and PyTorch are installed automatically. The trained ATLAS9, MARCS, and
PHOENIX weights are included, so interpolation works from any working directory
without downloading additional data. The package loads only its bundled legacy
PyTorch checkpoints, using `weights_only=False` because these contain training
metadata as well as model weights. Only replace these files with trusted checkpoints.

## Python usage

```python
from innterpol import innterpol

# [M/H], [C/M], [O/M], effective temperature (K), log(g)
atmosphere = innterpol([0, 0, 0, 5500, 2.5])
# Shape (71, 4): log mass, log temperature, log pressure, log electron density.
```

The other atmosphere grids have separate model implementations:

```python
from innterpol.marcs import innterpol as interpolate_marcs
from innterpol.phoenix import innterpol as interpolate_phoenix

marcs_atmosphere, marcs_tau = interpolate_marcs([0, 0, 0, 5500, 2.5])
phoenix_atmosphere, phoenix_tau = interpolate_phoenix([0, 0, 5500, 2.5])
# PHOENIX inputs: [M/H], [Alpha/M], Teff (K), log(g).
# Atmosphere shapes: (56, 4) for MARCS, (64, 4) for PHOENIX.
```

The input conventions, normalization, and output values follow the original
scripts; see the grid-specific READMEs for scientific details. For ATLAS9 the
optical depth coordinate is `numpy.arange(-6.875, 2, 0.125)`.

For plotting, install `python -m pip install ".[plot]"` and use Matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.arange(-6.875, 2, 0.125), 10 ** atmosphere[:, 1])
plt.xlabel("log optical depth")
plt.ylabel("Temperature (K)")
```

The former root `innterpol.py` entry point is now `example_atlas9.py`, to avoid
shadowing the installed `innterpol` package. MARCS and PHOENIX script entry points
remain available after installation. Training code and the older PCA12 experiment
remain standalone research scripts and are excluded from the installed package.
PCA12 requires SciPy, Matplotlib, and its separately downloaded coefficient file;
see [its README](iNNterpol_PCA12/README.md).

## Building and testing

```sh
python -m pip install build twine pytest
python -m build
python -m twine check dist/*
python -m pip install dist/*.whl
python -m pytest tests
```

Both the wheel and source distribution include all six trained checkpoint files.
The automated checks install from the source distribution and exercise all three
grids from an unrelated directory.

## Training

For full disclosure, all the numpy compressed data, the models (CAE and NN) and the programs to do the complete training are available. 
See https://github.com/cwestend/iNNterpol/tree/main/training_MARCS (for training on the MARCS grid of models). 

