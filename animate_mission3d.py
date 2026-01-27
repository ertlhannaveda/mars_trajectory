import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D
from mission import (
    hohmann_guess,
    hohmann_launch_time,
    make_initial_state
)
from integrators import integrate
from dynamics_mission import derivs
from bodies import earth_state, mars_state
from constants import MU_MARS

PARAMS_PATH = Path("last_optimized.json")

def load_params():
    """Load (t0, dv, alpha) from last_optimized.json; fallback to Hohmann."""
    if PARAMS_PATH.exists():
        data = json.loads(PARAMS_PATH.read_text())
        t0 = float(data["t0"])
        dv = float(data["dv"])
        alpha = float(data.get("alpha", 0.0))
        return t0, dv, alpha

    #fallback to Hohmann guess if file doesn't exist
    dv_hoh, _ = hohmann_guess()
    t0 = float(hohmann_launch_time(t_ref=0.0))
    return t0, float(dv_hoh), 0.0

#helper to keep main simulation loop clean
def planet_pos_vel(state_fn, t): #where planet is at time t
    r, v = state_fn(t)
    return np.asarray(r, dtype=float), np.asarray(v, dtype=float)

#core simulation function
def simulate_heliocentric(t0, dv, alpha, dt, tf, method):
    s0 = make_initial_state(t0=t0, dv=dv, alpha=alpha)
    ts, ys = integrate(derivs, s0, t0, t0 + tf, dt, method=method)

    r_sc = ys[:, 0:2]
    v_sc = ys[:, 2:4]
    r_e = np.empty_like(r_sc) #store EM
    r_m = np.empty_like(r_sc)
    v_m = np.empty_like(v_sc)

    for i, t in enumerate(ts): #where are planets at exact time
        r_e[i], _ = planet_pos_vel(earth_state, t)
        r_m[i], v_m[i] = planet_pos_vel(mars_state, t)

    d_m = np.linalg.norm(r_sc - r_m, axis=1) #distence
    idx_hit = int(np.argmin(d_m)) #smallest distence
    return ts, ys, idx_hit, float(d_m[idx_hit]), float(np.linalg.norm(v_sc[idx_hit] - v_m[idx_hit])), float(ts[idx_hit]), r_e, r_m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="last_optimized.json")
    ap.add_argument("--dt", type=float, default=0.005) #time step
    ap.add_argument("--method", default="rk4")
    ap.add_argument("--tf_factor", type=float, default=1.35)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    t0, dv, alpha = load_params()
    
    #get transfer time for simulation limit
    _, T_transfer = hohmann_guess()
    
    ts, ys, idx_hit, miss, vrel, t_hit, r_e, r_m = simulate_heliocentric(
        t0, dv, alpha, args.dt, args.tf_factor * T_transfer, args.method
    )

    #prepare Data
    r_sc_h, r_e_h, r_m_h = ys[:idx_hit+1, 0:2], r_e[:idx_hit+1], r_m[:idx_hit+1]
    r_sc_cap, r_m_cap = (None, None)

    #3D Plot Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 0.3]) 

    th = np.linspace(0, 2*np.pi, 500)
    ax.plot(np.linalg.norm(r_e_h[0])*np.cos(th), np.linalg.norm(r_e_h[0])*np.sin(th), 0, lw=0.7, ls='-', color='green', alpha=0.3)
    ax.plot(np.linalg.norm(r_m_h[0])*np.cos(th), np.linalg.norm(r_m_h[0])*np.sin(th), 0, lw=0.7, ls='-', color='red', alpha=0.3)
    ax.scatter([0], [0], [0], color='orange', s=100, label="Sun")

    ax.scatter([r_e_h[0,0]], [r_e_h[0,1]], [0], s=20, color='gray')
    ax.text(r_e_h[0,0], r_e_h[0,1], 0.1, "  Departure", fontsize=9, fontweight='bold')

    ax.scatter([r_m_h[-1,0]], [r_m_h[-1,1]], [0], s=80, color='red', marker='x')
    ax.text(r_m_h[-1,0], r_m_h[-1,1], 0.15, f"  Encounter\n  Miss: {miss:.4g} AU", 
            fontsize=9, fontweight='bold', color='darkred')


    traj_line, = ax.plot([], [], [], lw=2, color='navy', label="Spacecraft")
    sc_dot = ax.scatter([], [], [], s=30, color='black')
    e_dot  = ax.scatter([], [], [], s=50, color='green', edgecolors='white', label="Earth")
    m_dot  = ax.scatter([], [], [], s=50, color='red', edgecolors='white', label="Mars")

    info = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray"))

    ax.set_xlabel("X [AU]"); ax.set_ylabel("Y [AU]"); ax.set_zlim(-0.2, 0.2)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.05, 1))


    step = max(1, len(r_sc_h) // 150)
    frames_h = list(range(0, len(r_sc_h), step))
    
    def update(k):
        if k < len(frames_h):
            i = frames_h[k]
            traj_line.set_data(r_sc_h[:i+1, 0], r_sc_h[:i+1, 1])
            traj_line.set_3d_properties(np.zeros(i+1))
            sc_dot._offsets3d = ([r_sc_h[i,0]], [r_sc_h[i,1]], [0])
            e_dot._offsets3d = ([r_e_h[i,0]], [r_e_h[i,1]], [0])
            m_dot._offsets3d = ([r_m_h[i,0]], [r_m_h[i,1]], [0])
            info.set_text(f"Phase: Transfer\nTime: {ts[i]:.2f}\nDist: {np.linalg.norm(r_sc_h[i]-r_m_h[i]):.4g} AU")
        return ()

if __name__ == "__main__":
    main()