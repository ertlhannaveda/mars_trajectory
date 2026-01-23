import numpy as np
from constants import MU_SUN, MU_EARTH, MU_MARS
from bodies import earth_state, mars_state

def acceleration_spacecraft(t, r):      #gravitational acceleration on body
    """Compute acceleration at position r."""
    r = np.array(r, dtype=float)           #ensure spacecraft position is a numpy array
    a = np.zeros(2, dtype=float)
    eps = 0 #1e-10  # Very small softening parameter, so we cnat fly into earth or mars causing the code to crash (blunts gravity, slightly shifts the physics)

    # Sun
    d = r       # r in [x,y]; vector form sun to the spacecraft d = r_sc - r_S = r
    a -= MU_SUN * d / (np.linalg.norm(d)**3 + eps)      #newtons law of gravitation, pointing towards the sun

    # Earth
    rE, _ = earth_state(t)      #gets the position of earth at t; earths velocity ignored, not needed
    dE = r - rE                 #vector from earth to spacecraft; direction of pull from earth
    a -= MU_EARTH * dE / (np.linalg.norm(dE)**2 + eps)**1.5      #also newton, but for earth

    # Mars
    rM, _ = mars_state(t)       #the same as above but for mars
    dM = r - rM
    a -= MU_MARS * dM / (np.linalg.norm(dM)**2 + eps)**1.5

    return a        #[ax, ay]

def derivs(t, state):       #defines first order ODE for the integrators eg d/dt[r, v]= [v,a]
    """state = [x, y, vx, vy]"""
    r = np.array(state[:2], dtype=float)          #extracts position
    v = np.array(state[2:], dtype=float)           #extracts velocity

    a = acceleration_spacecraft(t, r)       #computes acceleration at that time 

    return np.hstack((v, a))                #combines velocity and acceleration into [vx, vy, ax, ay]
