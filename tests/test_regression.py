#!/usr/bin/env python3
import pytest
import numpy as np
import os
from lymanforest.pipeline import process_snapshot

def test_pipeline_regression():
    # 1. Define the paths
    current_dir = os.path.dirname(__file__)
    snap_path = os.path.join(current_dir, "data/test_snap.hdf5")
    # File previously generated, to store the right output
    expected_output_path = os.path.join(current_dir, "data/expected_flux.npy")
    
    # 2. QSO mock data
    qso_wl = np.linspace(3000, 6000, 100)
    qso_flux = np.ones(100)
    
    # 3. Execution
    results = process_snapshot(snap_path, qso_wl, qso_flux)
    final_flux = results[4] # final_qso_flux
    
    # 4. Comparison
    expected_flux = np.load(expected_output_path)
    # Check that the output is similar to the previous saved
    np.testing.assert_allclose(final_flux, expected_flux, rtol=1e-5)
