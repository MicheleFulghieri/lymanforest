import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from scipy.interpolate import interp1d
from numba import njit, prange
from .utils import cubic_spline_kernel, get_hit_mask, get_skewer_property, hubble, optical_depth_filtered
from .constants import m_p, k_b, gamma, c, sigma_0 



# --- The pipeline function ---
def process_snapshot(snap_path, qso_wl=None, qso_flux=None):  # default to none to correct import the pack
    """
    Runs the entire pipeline: Loading -> Units -> Master Skewer -> Tau -> Forward Model
    """
    # --- A. Upload and constants ---
    with h5py.File(snap_path, 'r') as f:
        header       = dict(f['Header'].attrs)

        # store the relevant header attributes
        hubble_param      = header['HubbleParam']    # for the velocity calculation and for units conversion (simulations usually use coomoving coordinates)
        redshift          = header['Redshift']       # z = 2.464
        scale_factor      = header['Time']           # a = 0.2886 at this redshift
        Lbox              = header['BoxSize']        # 25000.0 kpc, box size of Illustris TNG (units of kpc / (h * a))
        omega_m           = header['Omega0']         # 0.3 (flat universe with only Lambda and matter)
        omega_l           = header['OmegaLambda']    # 0.7 (flat universe with only Lambda and matter)

        # conversion factors
        kpc_to_cm = header['UnitLength_in_cm']                 # 3.085678e+21 cm/kpc
        ten_solar_mass_to_g = header['UnitMass_in_g']          # 1.989e+33 g/Msun
        kms_to_cms = header['UnitVelocity_in_cm_per_s']        # 100000.0 (cm/s) / (km/s)

        # conversion of Lbox in physical cgs units
        Lbox = Lbox * kpc_to_cm * scale_factor / hubble_param

        # data reading and convertion in physical units
        coordinates        = f['PartType0/Coordinates'][:] * kpc_to_cm * scale_factor / hubble_param
        masses             = f['PartType0/Masses'][:] * ten_solar_mass_to_g / hubble_param
        metallicity        = f['PartType0/GFM_Metallicity'][:]
        density            = f['PartType0/Density'][:] * ten_solar_mass_to_g * hubble_param**2 / (kpc_to_cm**3 * scale_factor**3)
        velocity           = f['PartType0/Velocities'][:] * kms_to_cms * np.sqrt(scale_factor)
        internal_energy    = f['PartType0/InternalEnergy'][:] * kms_to_cms**2
        electron_abundance = f['PartType0/ElectronAbundance'][:]
        h_smoothing        = f['PartType0/SubfindHsml'][:] * kpc_to_cm * scale_factor / hubble_param
        HI_abundance       = f['PartType0/NeutralHydrogenAbundance'][:]

    Npart = len(masses)
    print(f"\nSuccessfully loaded {Npart} gas particles from the snap at z = {redshift:.2f}")



    #  --- B. Temperature determination ---

    # pyhisical values
    m_p = 1.6726 * 1e-24    # proton mass in g
    k_b = 1.3806 * 1e-16    # boltzmann constant in erg/K
    X_H = 0.76
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * electron_abundance)      # mean molecular weight in units of proton mass
    H_z_cgs = hubble_param * kms_to_cms / (1000 * kpc_to_cm)           # hubble_param*100 km/s/Mpc. H_z_cgs = 2.17^-20 s^-1


    # gas temperature
    gas_temperature = (2.0/3.0) * (mu *m_p /k_b) * internal_energy     # (16220782,) array with the gas particle temperature in K
    print(f"Mean gas temperature of the snap at z = {redshift:.2f}: {np.mean(gas_temperature):.3f}K")



    # --- C. Master skewer properties ---

    # set up for the master skewer
    master_T, master_nHI, master_vrad = [], [], []   # array initialization
    np.random.seed(42)
    Nbins     = 50
    N_skewers = 20

    # efficacy masses for the SPH calulation (out of the loop, to save computational time)
    m_eff_temp = masses * gas_temperature / density        # (16220782,) array with the effective masses
    mass_HI    = masses * HI_abundance * X_H               # (16220782,) array
    m_eff_vpec = masses * velocity[:, 2] / density         # (16220782,), only z component -> velocity[:, 2]


    for i in range(N_skewers):

      # generate a random ray for each stiched skewer -> different pattherns
      # OBS: (x, y) stay constant and z varies along the 3D box: all the skewers are // to the z ax
      xray = np.random.uniform(0, Lbox)
      yray = np.random.uniform(0, Lbox)
      ray_coords = (xray, yray)

      # SPH caluclations
      z_bins, T_skewer = get_skewer_property(coordinates, h_smoothing, m_eff_temp, ray_coords, Lbox, Nbins, progress=False )
      z_bins, rho_HI = get_skewer_property(coordinates, h_smoothing, mass_HI, ray_coords, Lbox, Nbins, progress=False)
      nHI = rho_HI / m_p                        # convert the mass density in density of atoms, dividing by the proton mass (the mass of the H atom)
      z_bins, vpec = get_skewer_property(coordinates, h_smoothing, m_eff_vpec, ray_coords, Lbox, Nbins, progress=False)

      ## radial velocities
      H_z = hubble(z=redshift, omega_m=omega_m, omega_l=omega_l, h=H_z_cgs)     # Hubble function calculated at the snapshot redishift
      offset = Lbox * i                                                         # distance of a pixel in the i -th box
      vrad = vpec + H_z * (offset + z_bins)                                     # add the peculiar velocities to the Hubble flow

      # append in the master arrays
      master_T.append(T_skewer)     # list [array_1, array_2, ..., array_20] of 20 array
      master_nHI.append(nHI)
      master_vrad.append(vrad)

    # concatenate: convert the lists into a single 1D numpy array of shape (Nbins * Nskewers) to have the LoS properties in a single object
    master_T     = np.concatenate(master_T)
    master_nHI   = np.concatenate(master_nHI)
    master_vrad  = np.concatenate(master_vrad)



    # --- D. Optical Depth and Flux ---

    # preliminar calculations
    vel_grid = np.linspace(np.min(master_vrad), np.max(master_vrad), 30000)    # sampling the velocities: (30 000,) array of equispaziated values
    b = np.sqrt(2 * k_b * master_T / m_p)                                      # Doppler parameter
    bin_size = Lbox / Nbins                                                    # size in cm of one bin of the skewer

    # Snapshot: tau for each sampled grid velocity bin (observed velocity)
    master_tau = optical_depth_filtered(vel_grid, master_vrad, master_nHI, b, bin_size, sigma_0, c)

    # Snapshot: trasmission
    master_F = np.exp(-master_tau)   # F = I(v)/I0 = e^-tau


    # --- E. Fit the map to the Quasar ---

    # redshifted Lya wavelength
    lambda_lya     = 1215.67                         # Angstrom, intrinsic Lya wavelength
    lambda_qso_lya = lambda_lya * (1 + redshift)     # Angstrom
    print(f"Redshifted Lya wavelength emitted by the QSO at z={redshift:.2f} as seen from the Earth: {lambda_qso_lya:.2f} Angstrom")

    # conversion of the velocity grid in a wavelength grid for the simulated forest -> the wavelength grid of the simulated Lyman-alpha forest
    lambda_obs = lambda_qso_lya * (1 - vel_grid / c)           #  to account for the blueshift of the gas falling away from the qso frame
    print(f"Forest range: {np.min(lambda_obs):.1f} - {np.max(lambda_obs):.1f} Angstrom")

    # move the QSO at the snapshot redshift (the template is at z=0)
    qso_wl_redshifted = qso_wl * (1 + redshift)

    # Interpolate the simulated synthetic flux transmission (sim_F_sorted) onto the original QSO wavelength grid (qso_wl)
    # sort the simulated wavelengths and their corresponding flux transmission: interpolation functions expect sorted x-coordinates.
    sort_idx_sim  = np.argsort(lambda_obs)       # argsort returns the indices that would sort an array in ascending order
    lambda_obs = lambda_obs[sort_idx_sim]
    sim_F_sorted  = master_F[sort_idx_sim]   # sort the synthetic fluxes according to the quasar wavelenghts

    # interpolation
    f_interp = interp1d(lambda_obs, sim_F_sorted, bounds_error=False, fill_value=1.0) # fill_value=1.0 means no absorption outside the simulated range, bounds_error=False prevents error for out-of-bounds values
    master_F_interpolated = f_interp(qso_wl_redshifted)

    # apply absorption: Multiply the original QSO flux by the interpolated transmission factor
    final_qso_flux = qso_flux * master_F_interpolated


    return  lambda_qso_lya, master_tau, master_F, qso_wl_redshifted, final_qso_flux, redshift
    # returns, in order: the reshifted Lya wl,  tau and F of the master skewer, the redshifted wl of the qso, its flux, the redshift 
