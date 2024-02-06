"""
Using shared memory parallelism
===============================

This examples shows how to use multiple threads for particle spectra. The
threadpool can also be used with LOS dust, <WIP>
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from unyt import Myr

from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.galaxy import Galaxy


# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the star formation and metal enrichment histories
metal_dist = ZDist.Normal(mean=0.01, sigma=0.005)
sfh = SFH.Constant(duration=100 * Myr)  # constant star formation
stars = ParametricStars(
    grid.log10age,
    grid.metallicity,
    sf_hist=sfh,
    metal_dist=metal_dist,
)

# Create stars object
n = 10000  # number of particles for sampling
stars = sample_sfhz(
    stars.sfzh,
    stars.log10ages,
    stars.log10metallicities,
    n,
    initial_mass=10**6,
)

# Create galaxy object
galaxy = Galaxy(stars=stars)

# Generates stellar incident spectra in serial
start = time.time()
serial_sed = galaxy.stars.get_particle_spectra_incident(grid, nthreads=1)
print(f"Serial spectra took {time.time() - start} seconds")

# Generates stellar incident spectra with 4 threads
start = time.time()
sed4 = galaxy.stars.get_particle_spectra_incident(grid, nthreads=4)
print(f"4 thread spectra took {time.time() - start} seconds")

# Generates stellar incident spectra with 8 threads
start = time.time()
sed8 = galaxy.stars.get_particle_spectra_incident(grid, nthreads=8)
print(f"8 thread spectra took {time.time() - start} seconds")

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog()

# Plot each Sed
ax.plot(serial_sed.lam, np.sum(serial_sed.lnu, axis=0), label="Serial")
ax.plot(sed4.lam, np.sum(sed4.lnu, axis=0), label="$N_{threads} = 4$")
ax.plot(sed8.lam, np.sum(sed8.lnu, axis=0), label="$N_{threads} = 8$")

plt.legend()
plt.show()
