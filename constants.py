import numpy as np

#gravitational strengths of each of our planets
MU_SUN   = 1.0           # mu = GM, gravitational parameter
MU_EARTH = 3.003e-6      # Earth / Sun mass ratio; earths mass expressed as some integer multiple of the suns mass
MU_MARS  = 3.213e-7      # same but with mars

# AU astronomical units, nondimensional
R_EARTH = 1.0
R_MARS  = 1.523

# one full revolution, Earths orbit, to maintain dimensionlessness
TWOPI = 2.0 * np.pi
