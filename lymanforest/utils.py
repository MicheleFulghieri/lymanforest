import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from scipy.interpolate import interp1d
from numba import njit, prange
from .constants import m_p, k_b, gamma, c, sigma_0 


#  --- Preliminary functions ---

# kernel function of the SPH
@njit
def cubic_spline_kernel(q):
    """
    Computes the cubic spline kernel components.
    Input: q (numpy array) = distance / h
    Output: w (numpy array) = kernel value (without 1/h^3 scaling)
    """
    sigma = 1.0 / np.pi             # numerical prefactor (not including 1/h^3)
    w = np.zeros_like(q)            # np.zeros_like(q): array of zeros with the same shape and type as the given array q

    # case 0 <= q < 1
    mask1 = (q >= 0) & (q < 1)       # mask: array of boolean values (True or False) -> apply the operation on the array only where the condition is verified
    w[mask1] = 1 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3

    # case 1 <= q < 2
    mask2 = (q >= 1) & (q < 2)
    w[mask2] = 0.25 * (2 - q[mask2])**3

    return w * sigma


# mask
@njit
def get_hit_mask(centers, R, ray_coords, Lbox):
    xray, yray = ray_coords

    xc = centers[:, 0]
    yc = centers[:, 1]

    # 1. Calculate absolute differences
    dx = np.abs(xc - xray)
    dy = np.abs(yc - yray)

    # 2. Apply Periodicitiy
    # If distance > 0.5*Lbox, the shortest path is across the boundary.
    # In a periodic box of side L, the maximum distance between two points can never exceed L/2
    # boolean indexing: filter the element of an array only where the condition in [] is satisfied (ex: dx[dx >= 0.5*Lbox] -= ...)
    dx[dx >= 0.5*Lbox] -= Lbox   # condtion: [points are closer by wrapping around the edge] -> subtract Lbox to their distance (no matter for negative dx: they will be squared below)
    dy[dy >= 0.5*Lbox] -= Lbox

    # 3. Calculate distance squared in the plane XY, trasversal to z
    dist_sq = dx**2 + dy**2

    # 4. Create mask
    mask = (dist_sq <= R**2)

    return mask 


# skewer properties
@njit   # the application of this function below, will be computationally expensive
def get_skewer_property(positions, h_lengths, masses, ray_coords, Lbox, Nbins, progress=True):
    xray, yray = ray_coords

    # 1. Get Mask
    hit_mask = get_hit_mask(positions, 2*h_lengths, ray_coords, Lbox)     # get_hit_mask with the square distance R^2 = (2*h_lengths)^2

    # 2. Select Active Data
    p_pos = positions[hit_mask]     # (1000, 3) array
    p_h   = h_lengths[hit_mask]
    p_m   = masses[hit_mask]

    # 3. Periodic Transversal Distance (dx, dy)

    # 3.1 calculate absolute differences
    dx = np.abs(p_pos[:, 0] - xray)
    dy = np.abs(p_pos[:, 1] - yray)

    # 3.2 apply periodicity
    dx[dx >= 0.5*Lbox] -= Lbox
    dy[dy >= 0.5*Lbox] -= Lbox

    # 3.3 squared transverse (with respect to the skewer along z) distance: minimum distance over the slice z
    dxy2 = dx**2 + dy**2
    # dxy2 = np.zeros(len(p_pos)) # Placeholder, replace with actual calculation

    # Initialize Density Array
    ray_density = np.zeros(Nbins) # discretize the skewer
    dz = Lbox / Nbins             # length of a bin along z

    # Loop: for each particle, compute the kernel weight over all interested Z bins
    for i in range(len(p_pos)):   # Removed tqdm
        z_p  = p_pos[i, 2]     # z coordinate of the center of each of the i of the Npart particles
        h    = p_h[i]          # smoothing length of the i-th particle
        mass = p_m[i]          # mass of the i-th particle
        xy2  = dxy2[i]         # distace from the skewer of the i-th particle

        # h cannot be smaller than the bin in our calculation: the particle "cloud" is no smaller than a single skewer bin.
        # if a particle had a very small (very dense) h, it could fall between two bins without being counted, or create a mathematically infinite peak
        #  max(h, dz) ensure smooth sampling
        h_eff = max(h, dz)
        supp = 2.0 * h_eff      # R = 2h, support of the kernel function

        # Determine Z range of the kernel:  [zstart,zend], physical boundaries of the cloud along z
        z_start = z_p - supp
        z_end   = z_p + supp

        # Convert to raw integer indices (can be negative or > Nbins) i. e. labels the physical positions
        # Calculations such as bins (even negative ones) are covered by the particle (ex: If a particle is at
        # z=0.01 and its radius is 0.05, it covers the space from -0.04 to +0.06)
        # floor of x is: [x]:= the largest integer i, such that i <= x  (i.e. [2.56] = 2)
        idx_start = int(np.floor(z_start / dz))   # z_start / dz is the label of the start bin, except for a decimal part -> take the floor
        idx_end   = int(np.floor(z_end / dz))     # label of the last bin

        # Create array of indices involved
        idxs = np.arange(idx_start, idx_end + 1)

        # These are the physical positions of the bins, which might be outside [0, Lbox]
        # current_bin_zs using unwrapped idxs (e.g., -1, -2), to get the true physical position of the bins as if the box continued forever.
        # dz_sq = (current_bin_zs - z_p)**2 is hence calculated linearly and correctly.
        # Otherwise, closing the indices in the range [0, Nbins), the distance between a bin at 0.99 and a particle at 0.01 would have been 0.98 (wrong), instead of $0.02$.
        current_bin_zs = (idxs + 0.5) * dz    # label of the bin + 0.5 (to get the label of the center), then multplied for the z binning

        # Calculate 3D distance r and kernel q
        dz_sq = (current_bin_zs - z_p)**2    # squared difference between the zs coordinates of the skewer (current_bin_zs) and the z of the particle (z_p)
        r = np.sqrt(xy2 + dz_sq)             # summing under sqrt the squared difference of each coordinate: the 3D distance
        q = r / h_eff                        # the adimensional argument of the kernel (by def)
        norm = 1.0 / (h_eff**3)              # kernel's prefactor

        # Evaluate Kernel
        w = cubic_spline_kernel(q) * norm    # def of the kernel function

        # Wrap indices for the array: bridge between physics and the computer's RAM.
        # Use modulo operator (%) to handle periodic wrapping; modulo arithmetic: find the remainder after dividing one number by another 10 % 3 = 1
        # Index -1 becomes Nbins-1 (the last bin in the box). The mass that physically "extruded" from the left edge is mathematically added to the right edge.
        # Calculated the physics in an "open" (unwrapped) space to get the correct distances, but saved the data in a "closed" (wrapped) array to respect the periodicity.
        # 201 % 200 = 1: contribution that came out on the right comes back from the first position on the left
        # -1 % 200 = 199: The contribution that came out on the left comes back in from the last position on the right
        # projecting an infinite, periodic universe onto a finite memory. exiting on one side, then reappear on the other, maintaining your correct speed and relative position.
        wrapped_idxs = idxs % Nbins    # wrap the negative index -i back to the top of the array

        # Add to density
        ray_density[wrapped_idxs] += mass * w   # SPH formula of the density

    return (np.arange(Nbins) + 0.5) * dz, ray_density    # return the label of the position along z of the center of the z bin and the density contrinuting along the skewer


# Hubble function (LCDM flat universe)
def hubble(z, omega_m, omega_l, h):
  return 100 * h * np.sqrt(omega_m * (1+z)**3 + omega_l)     # 100 * h = H0


# optical depth calculation
@njit(parallel=True)
def optical_depth_filtered(vel_grid, vrad, nHI, b_array, dl, sigma_0=sigma_0, c=c):

  """ Sum the contributions of each gas bin of the master skewer to the optical depth array """

  # 1. initialization of the opacity array, with one entry for each sampled observed (grid) velocity
  tau = np.zeros_like(vel_grid)

  # determination of the constants out of the loop
  prefactor = sigma_0 * c / np.sqrt(np.pi)


  # 3. Double loop: over the grid of the velocity and a sum over the skewer binning
  for i in prange(len(vel_grid)):  # for each vgrid, the observed vel sampled, prange to exploit all the cores of the CPU

    v_obs = vel_grid[i]
    local_tau = 0.0         # initialize the scalar value of tau in the grid

    # 3.1 filter only the j-bin of the gas close to the sampled velocity
    for j in range(len(vrad)):

      current_b = b_array[j]       # get the scalar broadening parameter for this particle
      v_filter_j = 5 * current_b   # calculate filter for this particle

      # ignore the velocity out of the filter
      if np.abs(v_obs - vrad[j]) > v_filter_j:
        continue

      # if the radial velocity is in the filter (is closer than 5b to vgrid[i])
      local_tau += (nHI[j] * dl / current_b) * np.exp(- (v_obs - vrad[j])**2 / current_b**2)

    tau[i] = local_tau * prefactor
  return tau

