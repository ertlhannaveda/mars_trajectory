import numpy as np
import matplotlib as plt
from constants import MU_SUN, MU_EARTH, MU_MARS
from bodies import earth_state, mars_state

def total_energy(t, state):
    r = state[:2]
    v = state[2:]
    eps = 0 #1e-10
    
    kinetic = 0.5 * np.dot(v, v)
    
    # Sun
    potential = -MU_SUN / np.sqrt(np.dot(r, r) + eps)

    # Earth
    rE, _ = earth_state(t)
    dE = r - rE
    potential -= MU_EARTH / np.sqrt(np.dot(dE, dE) + eps)

    # Mars
    rM, _ = mars_state(t)
    dM = r - rM
    potential -= MU_MARS / np.sqrt(np.dot(dM, dM) + eps)

    return kinetic + potential