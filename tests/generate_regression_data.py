#!/usr/bin/env python3
import os
import numpy as np
from lymanforest.pipeline import process_snapshot

# Define the paths relative to the current directory
current_dir = os.path.dirname(__file__)
snap_path = os.path.join(current_dir, "data/test_snap.hdf5")
output_path = os.path.join(current_dir, "data/expected_flux.npy")

# Mock QSO data
wl = np.linspace(3000, 6000, 100)
flux = np.ones(100)

print(f"Execution of the pipeline on {snap_path}...")

# Execution f the pipeline on the test snap
res = process_snapshot("tests/data/test_snap.hdf5", wl, flux)

# Save the final flux (result at the index 4)
np.save("tests/data/expected_flux.npy", res[4])

print(f"Regression file successfully generated in: {output_path}")
