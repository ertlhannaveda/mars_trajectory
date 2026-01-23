import numpy as np

AU = 1.496e11          # meters
YEAR = 3.15576e7       # seconds

T_SCALE = YEAR / (2.0 * np.pi)  #time unit = 1year/2pi seconds; this way earths omega =1 & earths orbit theta(t)=t
V_SCALE = AU / T_SCALE          #velocity = 1 AU/nondimensional time unit

def to_nd_r(r_si):                  #conversion of r to nondimensional units from SI
    return np.array(r_si) / AU

def to_nd_v(v_si):                  #same for velocity
    return np.array(v_si) / V_SCALE

def from_nd_r(r_nd):                #take nondimensional vector r and turn it back into meters
    return np.array(r_nd) * AU

def from_nd_v(v_nd):                #same again for velocity
    return np.array(v_nd) * V_SCALE
#this way we can integrate with r ~= v ~= mu = 1 instead of the huge SI numbers in SI units