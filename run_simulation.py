from integrators import integrate
from dynamics import derivs
from analysis import total_energy
import matplotlib.pyplot as plt
import numpy as np

# 1. Stress Test Settings
OFFSET = 0.1               
V_ORBIT = 0.0              
y0 = np.array([1.0 + OFFSET, 0.0, 0.0, 1.0 + V_ORBIT], dtype=float)

# Increase dt to 0.05 so Leapfrog's 2nd-order wiggle becomes visible
# Increase tf to 1000 to allow RK4's non-symplectic drift to accumulate
t0, tf, dt = 0.0, 1000.0, 0.05 

methods = ["rk4", "leapfrog", "euler"]
results = {}

for m in methods:
    # Use your universal integrate interface
    T, Y = integrate(derivs, y0, t0, tf, dt, method=m)
    
    e_initial = total_energy(T[0], Y[0])
    # Relative Energy Error calculation
    rel_errors = [np.abs((total_energy(t, y) - e_initial) / e_initial) for t, y in zip(T, Y)]
    results[m] = (T, rel_errors)
    
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

for ax, m in zip(axes, methods):
    T, err = results[m]
    ax.plot(T, err, lw=0.5) # Use thinner lines to see the Leapfrog 'texture'
    ax.set_yscale('log')
    # Set a specific Y-limit so we can see the 'drift' vs 'wiggle'
    ax.set_ylim(1e-12, 1e0) 
    ax.set_title(f"Numerical Stability: {m.upper()}")
    ax.set_ylabel("Relative Energy Error")
    ax.grid(True, which="both", linestyle='--', alpha=0.3)

axes[-1].set_xlabel("Nondimensional Time")
plt.tight_layout()
plt.show()

for m in methods:
    print(f"{m} Max Error: {np.max(results[m][1]):.2e}")