import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from integrators import integrate
from dynamics_mission import derivs
from bodies import earth_state, mars_state
from mission import simulate_mission_with_events

PARAMS_PATH = Path("last_optimized.json")

DT = 0.002
METHOD = "rk4"
TF_FACTOR = 1.35

FPS = 12
SKIP = 10
TRAIL_LEN = 4500
PLANET_TRAIL = 250
PAD_AU = 0.25 #spatial padding around orbits
INTERVAL_MS = int(1000 / FPS) #temporal speed

#load parameters from json file
def load_params_strict(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required params file: {path.resolve()}")
    data = json.loads(path.read_text())
    for k in ("t0", "dv", "alpha"):
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path.resolve()}")
    miss_au = float(data["miss_au"]) if "miss_au" in data else None
    return {"t0": float(data["t0"]), "dv": float(data["dv"]), "alpha": float(data["alpha"]), "miss_au": miss_au}

def orbit_circle(radius, n=900): #generate circle points for orbits, resolution
    th = np.linspace(0.0, 2.0 * np.pi, n)#list of angles distributed around circle
    return radius * np.cos(th), radius * np.sin(th) #trig

def main():
    p = load_params_strict(PARAMS_PATH)
    t0, dv, alpha = p["t0"], p["dv"], p["alpha"]

    if p["miss_au"] is not None and np.isfinite(p["miss_au"]): #set trigger for when to swich from sg to mg
        capture_radius_au = max(5e-5, 1.25 * p["miss_au"])
    else:
        capture_radius_au = 1e-4

    #how long the simulation continues after in M orbit
    POST_CAPTURE_TIME = 1.0

    #call to run the mission sim
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

    cap_event = next((e for e in events if e.get("name") == "mars_capture_burn"), None) #search for specific moment engine fires
    has_capture = (T2 is not None) and (Y2 is not None) and (cap_event is not None) #safety check, only animate orbit if we have data

    #combine pre- and post-capture data -- E to M orbit
    if has_capture:
        if abs(float(T2[0]) - float(T1[-1])) < 1e-12:
            T_all = np.concatenate([T1, T2[1:]]) #avoid duplicate time entry at capture
            Y_all = np.vstack([Y1, Y2[1:]])
        else:
            T_all = np.concatenate([T1, T2])
            Y_all = np.vstack([Y1, Y2])
    else:
        T_all, Y_all = T1, Y1

  
    idx = np.arange(0, len(T_all), SKIP, dtype=int) #downsample for animation speed
    T = T_all[idx]
    Y = Y_all[idx]

    
    cap_start_anim = None
    if has_capture:
        t_ca = float(metrics["t_ca"]) #exact closest approach time
        cap_start_anim = int(np.searchsorted(T, t_ca, side="left")) #find closest frame index of arrival time
        cap_start_anim = int(np.clip(cap_start_anim, 0, len(T) - 1)) #safety clip

        # closest index in FULL timeline (pre-downsample)
        i_ca_all = int(np.searchsorted(T_all, t_ca, side="left")) #find closest frame index of arrival time
        i_ca_all = int(np.clip(i_ca_all, 1, len(T_all) - 1))
        burn_pos = np.array([Y_all[i_ca_all, 0], Y_all[i_ca_all, 1]], dtype=float) #store x,y ofspacecraft at that moment

    #update planet positions for animation
    earth_xy = np.array([earth_state(float(t))[0] for t in T], dtype=float)
    mars_xy = np.array([mars_state(float(t))[0] for t in T], dtype=float)

    #orbit circles
    rE = float(np.linalg.norm(earth_xy[0]))
    rM = float(np.linalg.norm(mars_xy[0]))
    Ex, Ey = orbit_circle(rE)
    Mx, My = orbit_circle(rM)

    rmax = max(rE, rM) + PAD_AU #set figure bounds

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.grid(True, alpha=0.10)
    ax.set_title("Heliocentric Earth → Mars (capture burn shown)")

    ax.plot(Ex, Ey, linewidth=1, color ='green',label="Earth orbit")
    ax.plot(Mx, My, linewidth=1,color ='red' ,label="Mars orbit")
    ax.scatter([0.0], [0.0], s=120, color ='orange', label="Sun")

    earth_pt = ax.scatter([], [],color ='green', s=60, label="Earth")
    mars_pt = ax.scatter([], [], s=75,color ='red' ,label="Mars")
    sc_pt = ax.scatter([], [], s=35, color ='black', label="Spacecraft")

    sc_trail, = ax.plot([], [], color ='black', linewidth=1, label="Spacecraft path")
    earth_trail, = ax.plot([], [],color ='green', linewidth=1, alpha=0.45)
    mars_trail, = ax.plot([], [], color ='red',linewidth=1, alpha=0.45)

    burn_pt = ax.scatter([], [], s=120,color ='red', marker="x", linewidths=2, label="Capture burn")
    burn_pt.set_alpha(0.0)

    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10)
    ax.legend(loc="best")

    def update(frame: int):
        xE, yE = earth_xy[frame]
        xM, yM = mars_xy[frame]
        xS, yS = Y[frame, 0], Y[frame, 1]

        earth_pt.set_offsets(np.array([[float(xE), float(yE)]], dtype=float))
        mars_pt.set_offsets(np.array([[float(xM), float(yM)]], dtype=float))
        sc_pt.set_offsets(np.array([[float(xS), float(yS)]], dtype=float))

        s0 = max(0, frame - TRAIL_LEN) 
        sc_trail.set_data(Y[s0:frame + 1, 0], Y[s0:frame + 1, 1])
        #s0,p0: how far back the trail goes
        p0 = max(0, frame - PLANET_TRAIL)
        earth_trail.set_data(earth_xy[p0:frame + 1, 0], earth_xy[p0:frame + 1, 1])
        mars_trail.set_data(mars_xy[p0:frame + 1, 0], mars_xy[p0:frame + 1, 1])
        #capture burn visualization
        if has_capture and (burn_pos is not None) and (cap_start_anim is not None):
            flash = abs(frame - cap_start_anim) <= 6 #flash for 6 frames around capture
            if frame >= cap_start_anim: 
                burn_pt.set_offsets(np.array([[burn_pos[0], burn_pos[1]]], dtype=float))
                burn_pt.set_alpha(1.0 if flash else 0.65)
                burn_pt.set_sizes([220 if flash else 120])
            else:
                burn_pt.set_alpha(0.0)
        else:
            burn_pt.set_alpha(0.0)

        dv_cap_txt = ""
        if has_capture and cap_event is not None and ("dv" in cap_event):
            dv_cap_txt = f"\nΔv_cap ≈ {float(cap_event['dv']):.3g}"

        hud.set_text(
            f"t = {float(T[frame]):.5f}\n"
            f"t0 = {t0:.5f}\n"
            f"dv = {dv:.6f}\n"
            f"alpha = {alpha:.6f}\n"
            f"miss(CA) ≈ {float(metrics['d_ca']):.3g} AU\n"
            f"capture_radius ≈ {capture_radius_au:.3g} AU\n"
            f"capture = {'YES' if has_capture else 'NO'}"
            + dv_cap_txt
        )
 
        return earth_pt, mars_pt, sc_pt, sc_trail, earth_trail, mars_trail, burn_pt, hud

    anim = FuncAnimation(fig, update, frames=len(T), interval=INTERVAL_MS, blit=False, repeat=True)
    plt.show()
    plt.show()

if __name__ == "__main__":
    main()
