import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from integrators import integrate
from dynamics_mission import derivs
from bodies import earth_state, mars_state
from mission import (
    hohmann_guess,
    hohmann_launch_time,
    simulate_mission_with_events,
)

PARAMS_PATH = Path("last_optimized.json")

DT = 0.002
METHOD = "rk4"
TF_FACTOR = 1.35

# Animation
FPS = 15  # Increased frames per second for smoother and quicker animation
SKIP = 10  # Further increased skip value to render even fewer frames
TRAIL_LEN = 7000       # spacecraft trail length (points)
PLANET_TRAIL = 220    # planet trail length (points)
PAD_AU = 0.25         # view padding
INTERVAL_MS = 100 / FPS  # Define INTERVAL_MS based on FPS for consistent animation speed


def load_params():
    """Load (t0, dv, alpha) from last_optimized.json; fallback to Hohmann."""
    if PARAMS_PATH.exists():
        data = json.loads(PARAMS_PATH.read_text())
        t0 = float(data["t0"])
        dv = float(data["dv"])
        alpha = float(data.get("alpha", 0.0))
        return t0, dv, alpha

    dv_hoh, _ = hohmann_guess()
    t0 = float(hohmann_launch_time(t_ref=0.0))
    return t0, float(dv_hoh), 0.0

#make geometry for perfecr circular orbit
def orbit_circle(radius, n=900):
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return radius * np.cos(th), radius * np.sin(th)


def main():
    t0, dv, alpha = load_params()

    #run mission sim (transfer + optional capture orbit)
    T1, Y1, T2, Y2, events, metrics = simulate_mission_with_events(
        integrate_fn=integrate,
        derivs_fn=derivs,
        t0=t0,
        dv=dv,
        alpha=alpha,
        dt=DT,
        tf_factor=TF_FACTOR,
        method=METHOD,
        do_capture=True,
        capture_radius_au=5e-5,
    )

    # Merge into one timeline for animation (EM and MS)
    if (T2 is not None) and (Y2 is not None):
        if abs(float(T2[0]) - float(T1[-1])) < 1e-12: #check if there is capture orbit
            T = np.concatenate([T1, T2[1:]]) #join time and state arrays (1: if times overlap skip first element of second phase)
            Y = np.vstack([Y1, Y2[1:]])
        else:
            T = np.concatenate([T1, T2])
            Y = np.vstack([Y1, Y2])
    else:
        T = T1
        Y = Y1

    #slicing/downsampling
    idx = np.arange(0, len(T), SKIP, dtype=int) #only keep ever 10th datapoint
    T = T[idx]
    Y = Y[idx]

    #precompute planet positions
    earth_xy = np.array([earth_state(float(t))[0] for t in T])
    mars_xy = np.array([mars_state(float(t))[0] for t in T])

    #orbit circles (use radii at launch time)
    rE = float(np.linalg.norm(earth_xy[0]))
    rM = float(np.linalg.norm(mars_xy[0]))
    Ex, Ey = orbit_circle(rE)
    Mx, My = orbit_circle(rM)

    #view limits
    rmax = max(rE, rM) + PAD_AU

    #key indices (closest approach computed on transfer arrays) need beacuse skip 10 frames, 
    i_ca_raw = int(metrics["i_ca"])
    i_ca_anim = int(i_ca_raw // SKIP) #divide raw index by skip value
    i_ca_anim = int(np.clip(i_ca_anim, 0, len(T) - 1))
    i_launch = 0

    cap = [e for e in events if e["name"] == "mars_capture_burn"]
    has_capture = bool(cap)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.grid(True, alpha=0.08)
    ax.set_title("2D Earth → Mars transfer)")

    #orbits
    ax.plot(Ex, Ey, color ='blue',linewidth=1, label="Earth orbit")
    ax.plot(Mx, My, color ='red',linewidth=1, label="Mars orbit")

    #bodies and spacecraft (animated points)
    sun_pt = ax.scatter([0.0], [0.0], color ='orange',s=120, label="Sun")
    earth_pt = ax.scatter([], [],color ='green', s=60, label="Earth")
    mars_pt = ax.scatter([], [], color ='red',s=75, label="Mars")
    sc_pt = ax.scatter([], [],color ='black', s=35, label="Spacecraft")

    #trails
    sc_trail, = ax.plot([], [],color ='gray', linewidth=1, label="Spacecraft path")
    earth_trail, = ax.plot([], [], linewidth=1, alpha=0.45)
    mars_trail, = ax.plot([], [], linewidth=1, alpha=0.45)

    #static event markers / labels
    ax.scatter(earth_xy[i_launch, 0], earth_xy[i_launch, 1], s=90)
    ax.text(
        earth_xy[i_launch, 0] + 0.03,
        earth_xy[i_launch, 1] + 0.03,
        "Launch",
        fontsize=10,
    )

    ax.scatter(mars_xy[i_ca_anim, 0], mars_xy[i_ca_anim, 1], s=95)
    ax.scatter(Y[i_ca_anim, 0], Y[i_ca_anim, 1], s=60)


    hud = ax.text( #create emplty bot in bottom left, put text there
        0.02, 0.98, "",
        transform=ax.transAxes, #attach to screen not map
        va="top", ha="left",
        fontsize=10
    )

    ax.legend(loc="best")

    #call for every frame
    def update(frame: int):
        xE, yE = earth_xy[frame]
        xM, yM = mars_xy[frame]
        xS, yS = Y[frame, 0], Y[frame, 1]

        earth_pt.set_offsets([xE, yE]) #slides points to new
        mars_pt.set_offsets([xM, yM])
        sc_pt.set_offsets([xS, yS])

        # trails
        s0 = max(0, frame - TRAIL_LEN)
        sc_trail.set_data(Y[s0:frame + 1, 0], Y[s0:frame + 1, 1])

        p0 = max(0, frame - PLANET_TRAIL)
        earth_trail.set_data(earth_xy[p0:frame + 1, 0], earth_xy[p0:frame + 1, 1])
        mars_trail.set_data(mars_xy[p0:frame + 1, 0], mars_xy[p0:frame + 1, 1])

        hud.set_text(
            f"t = {T[frame]:.5f}\n"
            f"t0 = {t0:.5f}\n"
            f"dv = {dv:.6f}, alpha = {alpha:.5f}\n"
            f"closest miss (transfer) ≈ {metrics['d_ca']:.3g} AU"
        )

        return earth_pt, mars_pt, sc_pt, sc_trail, earth_trail, mars_trail, hud

    anim = FuncAnimation(
        fig,
        update,
        frames=len(T),
        interval=INTERVAL_MS,
        blit=False,
        repeat=True,
    )

    plt.show()

if __name__ == "__main__":
    main()
