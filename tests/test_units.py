#!/usr/bin/env python3
import pytest
import numpy as np
from lymanforest.utils import hubble

def test_hubble_value():
    # Test the Hubble function with known values
    # If z=0, H(z) must be H0
    h0_cgs = 0.7 * 1e5 / (3.086e24) # un valore tipico
    res = hubble(z=0, omega_m=0.3, omega_l=0.7, h=h0_cgs)
    assert res == pytest.approx(h0_cgs)

def test_hubble_evolution():
    # Test that the expansion increase with z
    h0_cgs = 2.1e-18
    h_z1 = hubble(z=1, omega_m=0.3, omega_l=0.7, h=h0_cgs)
    h_z2 = hubble(z=2, omega_m=0.3, omega_l=0.7, h=h0_cgs)
    assert h_z2 > h_z1
