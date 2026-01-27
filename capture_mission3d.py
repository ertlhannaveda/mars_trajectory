# animate_mission_3d.py  (3D version of your 2D capture animation; z=0 plane)

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import PillowWriter

from integrators import integrate
from dynamics_mission import derivs
from bodies import earth_state, mars_state
from mission import simulate_mission_with_events

 
# Path to the JSON file containing optimized mission parameters
PARAMS_PATH = Path("last_optimized.json")

# Simulation parameters
DT = 0.002  # Time step for integration
METHOD = "rk4"  # Integration method (Runge-Kutta 4th order)
TF_FACTOR = 1.35  # Factor to extend simulation time

# Animation parameters
FPS = 12  # Frames per second
SKIP = 10  # Downsampling factor for animation frames
TRAIL_LEN = 4500  # Length of the spacecraft trail
PLANET_TRAIL = 250  # Length of the planet trails
PAD_AU = 0.25  # Padding around the orbits in AU
INTERVAL_MS = int(1000 / FPS)  # Interval between animation frames in milliseconds

# Function to load mission parameters from a JSON file
def load_params_strict(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required params file: {path.resolve()}")
    data = json.loads(path.read_text())
    for k in ("t0", "dv", "alpha"):
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path.resolve()}")
    miss_au = float(data["miss_au"]) if "miss_au" in data else None
    return {"t0": float(data["t0"]), "dv": float(data["dv"]), "alpha": float(data["alpha"]), "miss_au": miss_au}

# Function to generate a circular orbit for visualization
def orbit_circle(radius, n=900):
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return radius * np.cos(th), radius * np.sin(th)

# Main function to simulate and animate the mission
def main():
    # Load mission parameters
    p = load_params_strict(PARAMS_PATH)
    t0, dv, alpha = p["t0"], p["dv"], p["alpha"]

    # Ensure capture triggers for the saved solution
    if p["miss_au"] is not None and np.isfinite(p["miss_au"]):
        capture_radius_au = max(5e-5, 1.25 * p["miss_au"])
    else:
        capture_radius_au = 1e-4

    POST_CAPTURE_TIME = 1.0  # Time to simulate after capture

    # Simulate the mission with events
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
        capture_radius_au=capture_radius_au,
        post_capture_time=POST_CAPTURE_TIME,
    )

    # Check if capture occurred
    cap_event = next((e for e in events if e["name"] == "mars_capture_burn"), None)
    has_capture = (T2 is not None) and (Y2 is not None) and (cap_event is not None)

    # Combine pre- and post-capture segments
    if has_capture:
        if abs(float(T2[0]) - float(T1[-1])) < 1e-12:
            T_all = np.concatenate([T1, T2[1:]])
            Y_all = np.vstack([Y1, Y2[1:]])
        else:
            T_all = np.concatenate([T1, T2])
            Y_all = np.vstack([Y1, Y2])
    else:
        T_all, Y_all = T1, Y1

    # Downsample by index (safe now because dt is consistent)
    idx = np.arange(0, len(T_all), SKIP, dtype=int)
    T = T_all[idx]
    Y = Y_all[idx]

    # Capture start frame (by time)
    cap_start_anim = None
    if has_capture:
        t_ca = float(metrics["t_ca"])
        cap_start_anim = int(np.searchsorted(T, t_ca, side="left"))
        cap_start_anim = int(np.clip(cap_start_anim, 0, len(T) - 1))

    # Planets computed from T (they WILL keep moving)
    earth_xy = np.array([earth_state(float(t))[0] for t in T], dtype=float)
    mars_xy = np.array([mars_state(float(t))[0] for t in T], dtype=float)

    # Orbit circles for context
    rE = float(np.linalg.norm(earth_xy[0]))
    rM = float(np.linalg.norm(mars_xy[0]))
    Ex, Ey = orbit_circle(rE)
    Mx, My = orbit_circle(rM)

    rmax = max(rE, rM) + PAD_AU

    # --- 3D Figure setup ---
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("Heliocentric Earth → Mars (capture, 3D view; z=0)")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")

    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)

    # Give a small z-range so it *looks* 3D, but everything is still on z=0.
    zpad = 0.15 * rmax
    ax.set_zlim(-zpad, zpad)

    # Orbits at z=0
    ax.plot(Ex, Ey, np.zeros_like(Ex), linewidth=1, label="Earth orbit")
    ax.plot(Mx, My, np.zeros_like(Mx), linewidth=1, label="Mars orbit")
    ax.scatter([0.0], [0.0], [0.0], s=120, label="Sun")

    # Bodies as 3D scatters
    earth_pt = ax.scatter([], [], [], s=60, label="Earth")
    mars_pt = ax.scatter([], [], [], s=75, label="Mars")
    sc_pt = ax.scatter([], [], [], s=35, label="Spacecraft")

    # Trails as 3D lines
    sc_trail, = ax.plot([], [], [], linewidth=1, label="Spacecraft path")
    earth_trail, = ax.plot([], [], [], linewidth=1, alpha=0.45)
    mars_trail, = ax.plot([], [], [], linewidth=1, alpha=0.45)

    # HUD (2D overlay)
    hud = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="best")

    # Function to update the animation frame
    def update(frame: int):
        xE, yE = earth_xy[frame]
        xM, yM = mars_xy[frame]
        xS, yS = Y[frame, 0], Y[frame, 1]

        # everything stays on z=0
        zE = zM = zS = 0.0

        # Update points (3D scatter uses _offsets3d)
        earth_pt._offsets3d = ([float(xE)], [float(yE)], [zE])
        mars_pt._offsets3d = ([float(xM)], [float(yM)], [zM])
        sc_pt._offsets3d = ([float(xS)], [float(yS)], [zS])

        # Spacecraft trail
        s0 = max(0, frame - TRAIL_LEN)
        xs = Y[s0:frame + 1, 0]
        ys = Y[s0:frame + 1, 1]
        zs = np.zeros_like(xs)

        sc_trail.set_data(xs, ys)
        sc_trail.set_3d_properties(zs)

        # Planet trails
        p0 = max(0, frame - PLANET_TRAIL)

        xe = earth_xy[p0:frame + 1, 0]
        ye = earth_xy[p0:frame + 1, 1]
        earth_trail.set_data(xe, ye)
        earth_trail.set_3d_properties(np.zeros_like(xe))

        xm = mars_xy[p0:frame + 1, 0]
        ym = mars_xy[p0:frame + 1, 1]
        mars_trail.set_data(xm, ym)
        mars_trail.set_3d_properties(np.zeros_like(xm))

        # Optional camera rotation so it feels “3D”
        ax.view_init(elev=25, azim=30 + 0.25 * frame)

        hud.set_text(
            f"t = {float(T[frame]):.5f}\n"
            f"t0 = {t0:.5f}\n"
            f"dv = {dv:.6f}\n"
            f"alpha = {alpha:.6f}\n"
            f"miss(CA) ≈ {metrics['d_ca']:.3g} AU\n"
            f"capture_radius ≈ {capture_radius_au:.3g} AU\n"
            f"capture = {'YES' if has_capture else 'NO'}"
            + (f"\nΔv_cap ≈ {cap_event['dv']:.3g}" if has_capture else "")
        )

        return earth_pt, mars_pt, sc_pt, sc_trail, earth_trail, mars_trail, hud

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(T), interval=INTERVAL_MS, blit=False, repeat=True)
    plt.show()
    print("Saving GIF... this may take a minute.")
    writer = PillowWriter(fps=FPS) # Matches your animation speed
    anim.save("capture_mission3d.gif", writer=writer)

if __name__ == "__main__":
    main()
