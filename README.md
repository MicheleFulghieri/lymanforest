# lymanforest

A Python package to process Lyman-alpha forest snapshots from cosmological simulations (e.g., Illustris TNG).

## Overview
`lymanforest` provides a streamlined pipeline to transform raw simulation data into synthetic Quasar (QSO) spectra. It includes tools for:
* Physical units conversion for TNG snapshots.
* Gas temperature determination.
* SPH interpolation for Master Skewer properties.
* Optical depth and flux transmission calculations.

## Installation

### From Test-PyPI
You can install the latest version of the package using:
```bash
pip install --index-url [https://test.pypi.org/simple/](https://test.pypi.org/simple/) --extra-index-url [https://pypi.org/simple](https://pypi.org/simple) lymanforest


# Quick start
<from lymanforest.pipeline import process_snapshot

# Run the pipeline
results = process_snapshot(
    snap_path='path/to/snapshot.hdf5',
    qso_wl=qso_wl_array, 
    qso_flux=qso_flux_array
)

# results contains: lambda_qso_lya, master_tau, master_F, qso_wl_redshifted, final_qso_flux, redshift
