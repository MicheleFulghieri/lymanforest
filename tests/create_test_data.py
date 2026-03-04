#!/usr/bin/env python3
import h5py
import numpy as np
import os

def create_dummy_snap(output_path="tests/data/test_snap.hdf5"):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # --- A. Header creation ---
        header = f.create_group("Header")
        header.attrs['HubbleParam'] = 0.7
        header.attrs['Redshift'] = 2.44
        header.attrs['Time'] = 1.0 / (1.0 + 2.44)
        header.attrs['BoxSize'] = 25000.0
        header.attrs['Omega0'] = 0.3
        header.attrs['OmegaLambda'] = 0.7
        header.attrs['UnitLength_in_cm'] = 3.085678e21
        header.attrs['UnitMass_in_g'] = 1.989e33
        header.attrs['UnitVelocity_in_cm_per_s'] = 100000.0

        # --- B. PartType0 object creation (Gas) ---
        n_part = 100  # few particles are sufficient for the test
        gas = f.create_group("PartType0")
        
        # Coordinates (distributed in 0-25000 box)
        gas.create_dataset("Coordinates", data=np.random.rand(n_part, 3) * 25000.0)
        # Masses (random values around 0.01 in TNG units)
        gas.create_dataset("Masses", data=np.ones(n_part) * 0.01)
        # Metallicity
        gas.create_dataset("GFM_Metallicity", data=np.ones(n_part) * 0.02)
        # Density
        gas.create_dataset("Density", data=np.random.rand(n_part) * 0.1)
        # Velocity (km/s)
        gas.create_dataset("Velocities", data=np.random.randn(n_part, 3) * 100.0)
        # Internal energy
        gas.create_dataset("InternalEnergy", data=np.ones(n_part) * 1000.0)
        # Electron abundance
        gas.create_dataset("ElectronAbundance", data=np.ones(n_part) * 1.15)
        # Smoothing length (Hsml)
        gas.create_dataset("SubfindHsml", data=np.ones(n_part) * 50.0)
        # Neutral Hydrogen
        gas.create_dataset("NeutralHydrogenAbundance", data=np.random.rand(n_part) * 0.01)

    print(f"Test file sucessfully created in: {output_path}")

if __name__ == "__main__":
    create_dummy_snap()
