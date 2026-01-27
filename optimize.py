import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from integrators import integrate
from dynamics_mission import derivs
from mission import (
    hohmann_guess,
    hohmann_launch_time,
    simulate_mission_with_events,
)

# If set, overrides the Hohmann-phasing launch time.
T0_START = None

# Optimize launch time offset (dt0) as a decision variable.
OPTIMIZE_T0 = True

# Integrator settings (keep dt modest for speed; final run uses a smaller dt).
METHOD = "rk4"
DT_OBJECTIVE = 0.002
DT_FINAL = 0.0015
TF_FACTOR = 1.35

# Desired closest-approach miss distance (AU). Your target ~1e-4 AU.
TARGET_MISS_AU = 1e-4

OUT_PATH = Path("last_optimized.json")


def _base_launch_time() -> float: # Get base launch time t0.
    return float(T0_START) if T0_START is not None else float(hohmann_launch_time(t_ref=0.0))

# Run a mission simulation once with given parameters.
def _simulate(t0: float, dv: float, alpha: float, *, dt: float):
    """Run a mission simulation and return (T1, Y1, events, metrics)."""
    T1, Y1, _, _, events, metrics = simulate_mission_with_events(
        integrate_fn=integrate,
        derivs_fn=derivs,
        t0=float(t0),
        dv=float(dv),
        alpha=float(alpha),
        dt=float(dt),
        tf_factor=float(TF_FACTOR),
        method=METHOD,
        do_capture=False,  # optimization phase: just intercept Mars (capture handled later)
    )
    """
    returns:
    T1: times
    Y1: states [x,y,vx,vy]
    events: launch + closest approach markers
    metrics: d_window, vrel_hit, d_ca, etc.
    """

    return T1, Y1, events, metrics


def objective(x: np.ndarray) -> float:
    """Objective: reach Mars with small miss distance and reasonable v_rel.
    Returns a single scalar “badness” score for the optimizer to minimize."""
    dv = float(x[0])
    alpha = float(x[1])

    t0_base = _base_launch_time() #optimizing launch time 
    if OPTIMIZE_T0:
        dt0 = float(x[2]) #offset from base launch time
        t0 = t0_base + dt0
    else:
        t0 = t0_base #otherwize fixed

    try: #simulate mission if any error return large penalty
        T, Y, _, metrics = _simulate(t0, dv, alpha, dt=DT_OBJECTIVE)
    except Exception:
        return 1e9

    if np.any(~np.isfinite(Y)): #if NaN or inf return large penalty
        return 1e9

    r = np.linalg.norm(Y[:, :2], axis=1) #penalty for going too far from sun
    if np.any(r > 20.0) or np.any(r < 0.2):
        return 1e9

    miss = float(metrics["d_window"])  # use windowed miss distance, no cheating!
    vrel = float(metrics["vrel_hit"]) if np.isfinite(metrics["vrel_hit"]) else float(metrics["vrel_ca"])

    if not np.isfinite(miss) or miss == float("inf"): #if window score fail penalize
        return 1e9
    
    miss_term = ((miss / TARGET_MISS_AU) - 1.0) ** 2 #hit tarhet miss term
    far_penalty = (miss / 0.01) ** 2  # 0.01 AU is a very large miss for this toy problem

    dv1, _ = hohmann_guess() #encorage dv near hohmann dv1
    dv_reg = ((dv - dv1) / max(abs(dv1), 1e-6)) ** 2 #normalized dv to make scale reasonable
    alpha_reg = (alpha / 0.5) ** 2 #encorage alpha not to be huge
    vrel_term = (vrel / 0.2) ** 2 #encourage low vrel at encounter

    return float(20.0 * miss_term + 2.0 * far_penalty + 0.5 * vrel_term + 0.2 * dv_reg + 0.1 * alpha_reg)


def main():
    dv1, Ttr = hohmann_guess() 
    t0_base = _base_launch_time()

    print("Base launch time t0_base =", float(t0_base)) #just for debug
    print("Hohmann dv1 =", float(dv1), "  Hohmann ToF =", float(Ttr))

    if OPTIMIZE_T0:
        x0 = np.array([dv1, 0.0, 0.0], dtype=float) #start here
        bounds = [ #have to constrain search so it does not go insane
            (0.6 * dv1, 1.6 * dv1),  # dv
            (-0.7, 0.7),             # alpha (rad)
            (-0.8, 0.8),             # dt0 (time units)
        ]
    else:
        x0 = np.array([dv1, 0.0], dtype=float)
        bounds = [
            (0.6 * dv1, 1.6 * dv1),
            (-0.7, 0.7),
        ]

    res = minimize( #powell method: derivative-free, good for noisy obj0tives
        objective,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 260, "disp": True} #maxiter:limit iterations, disp:print progress
    )

    if OPTIMIZE_T0: #exact best params
        dv_opt, alpha_opt, dt0_opt = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
        t0 = float(t0_base + dt0_opt)
    else:
        dv_opt, alpha_opt = (float(res.x[0]), float(res.x[1]))
        t0 = float(t0_base)

    # Final higher-resolution run (still heliocentric intercept only)
    T, Y, events, metrics = _simulate(t0, dv_opt, alpha_opt, dt=DT_FINAL)
    miss = float(metrics["d_window"])
    t_hit = float(metrics["t_hit"])
    vrel = float(metrics["vrel_hit"]) if np.isfinite(metrics["vrel_hit"]) else float(metrics["vrel_ca"])

    print("\n=== Optimized mission parameters ===")
    print("t0_base =", float(t0_base))
    print("t0      =", float(t0))
    print("dv      =", float(dv_opt))
    print("alpha   =", float(alpha_opt))
    print("miss AU =", float(miss))
    print("vrel    =", float(vrel))
    print("t_hit   =", float(t_hit))
    print("objective =", float(res.fun))

    OUT_PATH.write_text( #write out results to json
        json.dumps(
            {"t0": float(t0), "dv": float(dv_opt), "alpha": float(alpha_opt), "miss_au": float(miss), "vrel": float(vrel)},
            indent=2,
        )
    )
    print(f"\nWrote {OUT_PATH}")

if __name__ == "__main__":
    main()
