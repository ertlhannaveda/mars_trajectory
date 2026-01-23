import numpy as np
from constants import MU_SUN, R_EARTH, R_MARS
#we use circular orbits for hohmann transfer
def circular_orbit(a, t, phase=0.0):            #a - radius of orbit, t- dimensionless time, phase - angular offset
    """Returns position and velocity for circular orbit."""
    omega = np.sqrt(MU_SUN / a**3)              #keplers third law, determines how fast a planet moves around the sun
    theta = omega * t + phase                   #angular position of the planet, phase is the initial angular offset

    r = np.array([a*np.cos(theta), a*np.sin(theta)])    #converts polar coordinates to cartesian vector in x-y plane
    v = np.array([-a*omega*np.sin(theta), a*omega*np.cos(theta)])       #veloctiy vector tangent to the orbit at all times, const magnitude
    return r, v     #position and velocity vector

def earth_state(t):         #fixed orbit, variable time
    return circular_orbit(R_EARTH, t)           #rE, vE = earth_state; are earths orbital radius and velocity

def mars_state(t):
    return circular_orbit(R_MARS, t)            #uses keplers law with a**3, so earth and mars have correct relative motion