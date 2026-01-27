import numpy as np

from constants import MU_SUN, MU_MARS, R_EARTH, R_MARS, TWOPI
from bodies import earth_state, mars_state

# Hohmann / phasing helpers
def hohmann_params(r1: float = R_EARTH, r2: float = R_MARS, mu: float = MU_SUN):
    """Classical Hohmann transfer parameters (2D, coplanar, heliocentric).

    Returns:
        dv1 : departure burn (tangential) relative to circular orbit at r1
        dv2 : arrival burn (tangential) relative to circular orbit at r2
        Ttr : half-period of transfer ellipse (time of flight)
        a_t : semi-major axis of transfer ellipse
        v1, v2 : circular speeds at r1, r2
        vt1, vt2 : transfer-ellipse speeds at r1, r2
    """
    a_t = 0.5 * (r1 + r2)

    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)

    vt1 = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
    vt2 = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))

    dv1 = vt1 - v1 #departure burn, add tangential speed to go from circular to transfer ellipse at r1
    dv2 = v2 - vt2 #arrival burn, match circular velocity
    Ttr = np.pi * np.sqrt((a_t ** 3) / mu) #half period of transfer ellipse, time of flight

    return float(dv1), float(dv2), float(Ttr), float(a_t), float(v1), float(v2), float(vt1), float(vt2)


def hohmann_guess():
    """Convenience return (dv1, Ttr) for Earth->Mars."""
    dv1, _, Ttr, *_ = hohmann_params()
    return float(dv1), float(Ttr)


def _mean_motion(a: float, mu: float = MU_SUN) -> float: #angular speed
    return float(np.sqrt(mu / (a ** 3)))


def hohmann_required_phase(T_transfer: float, r_target: float = R_MARS, mu: float = MU_SUN) -> float:
    """Required Mars-Earth phase angle (Mars ahead of Earth) for a Hohmann transfer."""
    nM = _mean_motion(r_target, mu=mu)
    # Classic result: phi = π - n_M * T_transfer (mod 2π)
    return float((np.pi - nM * float(T_transfer)) % TWOPI)


def hohmann_launch_time(
    t_ref: float = 0.0,
    phaseE: float = 0.0,
    phaseM: float = 0.0,
    mu: float = MU_SUN,
    rE: float = R_EARTH,
    rM: float = R_MARS,
) -> float:
    """Return the next launch time >= t_ref that satisfies Hohmann phasing."""
    dv1, Ttr = hohmann_guess() #Ttr: time of flight
    nE = _mean_motion(rE, mu=mu)
    nM = _mean_motion(rM, mu=mu)

    phi_req = hohmann_required_phase(Ttr, r_target=rM, mu=mu)

    denom = (nM - nE) #if the planets had same mean motion, relative phase would never change → you couldn’t “wait” for alignment.
    if abs(denom) < 1e-12:
        return float(t_ref)

    # (phaseM - phaseE) + (nM - nE) t == phi_req (mod 2π)
    t0_raw = (phi_req - (phaseM - phaseE)) / denom
    T_syn = TWOPI / abs(nE - nM) #synodic period,the time for the relative angle Δ(t) to advance by 2π.

    t0 = float(t0_raw) #pick the next solution; chossing k ao that t0 >= t_ref at the earliest
    if t0 < t_ref:
        k = np.ceil((t_ref - t0) / T_syn)
        t0 = t0 + float(k) * T_syn
    return float(t0)


# Basic geometry helpers
def rotate2(u: np.ndarray, alpha: float) -> np.ndarray: #takes 2d vector u and rotates it by angle alpha (radians)
    u = np.asarray(u, dtype=float) #convert u to numpy array of floats
    c, s = float(np.cos(alpha)), float(np.sin(alpha))
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]], dtype=float) 


def make_initial_state( #construct initial heliocentric spacecraft state near Earth at vector launch time
    t0: float,
    dv: float,
    alpha: float = 0.0,
    offset_au: float = 0.0,
    offset_dir: str = "tangential",
) -> np.ndarray:
    rE, vE = earth_state(t0)

    vE_norm = float(np.linalg.norm(vE)) #safety check; if earth position or velocity 0
    rE_norm = float(np.linalg.norm(rE))
    if vE_norm < 1e-15 or rE_norm < 1e-15:
        raise ValueError("Earth state is degenerate (norm too small).")

    t_hat = vE / vE_norm #unit tangent direction of earths orbit
    r_hat = rE / rE_norm #unit radial direction from sun to earth

    if str(offset_dir).lower().startswith("rad"): #decide how to offset
        offset_vec = float(offset_au) * r_hat 
    else:
        offset_vec = float(offset_au) * t_hat

    r0 = rE + offset_vec #final initial position
    inj_dir = rotate2(t_hat, float(alpha)) 
    v0 = vE + float(dv) * inj_dir

    return np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)


# Encounter / scoring
def closest_approach_to_mars(T: np.ndarray, Y: np.ndarray): 
    rM = np.array([mars_state(float(t))[0] for t in T], dtype=float)
    d = np.linalg.norm(Y[:, :2] - rM, axis=1)
    i = int(np.argmin(d))
    return float(d[i]), float(T[i]), i


def rendezvous_score(T: np.ndarray, Y: np.ndarray, t0: float, T_guess: float, window_frac: float = 0.10):
    """Find the best (smallest) spacecraft-Mars distance near the expected arrival time."""
    t_arr = float(t0) + float(T_guess) #expected arrival time
    t_lo = t_arr - float(window_frac) * float(T_guess) #create symmetric window
    t_hi = t_arr + float(window_frac) * float(T_guess)

    mask = (T >= t_lo) & (T <= t_hi) #check if if in arrival window
    if not np.any(mask):
        return float("inf"), float(t_arr), None

    Tw = T[mask] #time inside window
    Yw = Y[mask] #state inside window

    rM = np.array([mars_state(float(t))[0] for t in Tw], dtype=float) #for each time in window, get mars position
    d = np.linalg.norm(Yw[:, :2] - rM, axis=1) #distance to mars at each time in window

    j = int(np.argmin(d)) #index within window arrays
    i_best = int(np.where(mask)[0][j]) #convert to index within full arrays
    return float(d[j]), float(T[i_best]), i_best


def relative_state_at_index(T: np.ndarray, Y: np.ndarray, i: int): #helper to return position and velocity relative to mars
    t = float(T[int(i)])
    r_sc = Y[int(i), :2] #take first two columns of state row
    v_sc = Y[int(i), 2:]
    rM, vM = mars_state(t)

    r_rel = r_sc - rM #relative position and velocity vectors as seen from mars
    v_rel = v_sc - vM
    return r_rel, v_rel, float(np.linalg.norm(r_rel)), float(np.linalg.norm(v_rel))


def mission_metrics(T: np.ndarray, Y: np.ndarray, t0: float, T_guess: float, window_frac: float = 0.10):
    d_win, t_hit, i_hit = rendezvous_score(T, Y, t0, T_guess, window_frac=window_frac)
    if i_hit is not None:
        _, _, d_rel, v_rel = relative_state_at_index(T, Y, i_hit)
    else:
        d_rel, v_rel = float("inf"), float("inf")

    d_ca, t_ca, i_ca = closest_approach_to_mars(T, Y)
    _, _, _, vrel_ca = relative_state_at_index(T, Y, i_ca)

    return {
        "d_window": float(d_win),
        "t_hit": float(t_hit),
        "vrel_hit": float(v_rel),
        "d_ca": float(d_ca),
        "t_ca": float(t_ca),
        "i_hit": i_hit,
        "i_ca": int(i_ca),
        "vrel_ca": float(vrel_ca),
    }


# Burns / capture (Mars-centered)
def apply_instant_burn(state: np.ndarray, dv_vec: np.ndarray) -> np.ndarray:
    """Instantaneous burn on a heliocentric 2D state."""
    s = np.array(state, dtype=float)
    s[2:4] = s[2:4] + np.asarray(dv_vec, dtype=float)
    return s


def circular_orbit_velocity_about_mars(r_rel: np.ndarray) -> np.ndarray:
    """Tangential circular-orbit velocity (in the Mars-centered frame)."""
    r_rel = np.asarray(r_rel, dtype=float)
    r = float(np.linalg.norm(r_rel))
    if r <= 1e-12:
        return np.zeros(2, dtype=float)

    t_hat = np.array([-r_rel[1], r_rel[0]], dtype=float) / r
    v_circ = float(np.sqrt(MU_MARS / r))
    return v_circ * t_hat


def capture_burn_to_circular(T: np.ndarray, Y: np.ndarray, i_ca: int):
    """Compute a capture burn at closest approach to circularize about Mars (relative frame)."""
    t = float(T[int(i_ca)])
    rM, vM = mars_state(t)

    r_sc = Y[int(i_ca), :2]
    v_sc = Y[int(i_ca), 2:]

    r_rel = r_sc - rM
    v_rel = v_sc - vM

    v_rel_target = circular_orbit_velocity_about_mars(r_rel)
    dv_rel = v_rel_target - v_rel

    # Same dv applied in heliocentric frame changes relative velocity by dv_rel
    dv_helio = dv_rel
    return dv_helio, float(np.linalg.norm(dv_helio))


def mars_centered_derivs(t: float, state_rel: np.ndarray, soft_au: float = 1e-6) -> np.ndarray:
    """Two-body dynamics about Mars in the Mars-centered frame."""
    state_rel = np.asarray(state_rel, dtype=float)
    r = state_rel[:2]
    v = state_rel[2:]

    r2 = float(np.dot(r, r) + soft_au ** 2)
    inv = 1.0 / (r2 * np.sqrt(r2))
    a = -MU_MARS * r * inv
    return np.hstack((v, a))


def simulate_mission_with_events(
    integrate_fn,
    derivs_fn,
    t0: float,
    dv: float,
    alpha: float,
    *,
    dt: float = 0.002,
    tf_factor: float = 1.35,
    method: str = "rk4",
    offset_au: float = 0.0,
    offset_dir: str = "tangential",
    do_capture: bool = True,
    capture_radius_au: float = 5e-5,
    # NEW: how long to keep simulating after capture (in your time units)
    post_capture_time: float = 1.0,
):
    """
    Simulate heliocentric transfer and (optionally) continue heliocentric simulation after a capture burn.

    IMPORTANT: Post-capture is simulated in the SAME heliocentric dynamics (derivs_fn),
    so planets keep moving and the spacecraft can orbit Mars while both orbit the Sun.
    """
    dv_hoh, Ttr = hohmann_guess()
    tf = float(t0) + float(tf_factor) * float(Ttr)

    y0 = make_initial_state(t0, dv, alpha, offset_au=offset_au, offset_dir=offset_dir)
    T1, Y1 = integrate_fn(derivs_fn, y0, float(t0), float(tf), dt=float(dt), method=method)

    # encounter metrics
    d_hit, t_hit, i_hit = rendezvous_score(T1, Y1, t0, Ttr, window_frac=0.25)
    d_ca, t_ca, i_ca = closest_approach_to_mars(T1, Y1)
    _, _, _, vrel_ca = relative_state_at_index(T1, Y1, i_ca)

    metrics = mission_metrics(T1, Y1, t0, Ttr, window_frac=0.25)
    metrics.update({"Ttr": float(Ttr), "dv_hoh": float(dv_hoh)})

    events = [
        {"name": "launch_burn", "t": float(t0), "i": 0, "dv": float(dv)},
        {"name": "closest_approach", "t": float(t_ca), "i": int(i_ca), "miss_au": float(d_ca), "vrel": float(vrel_ca)},
    ]

    T2 = Y2 = None

    # --- optional capture (heliocentric continuation) ---
    if do_capture and (d_ca <= float(capture_radius_au)):
        dv_cap_vec, dv_cap = capture_burn_to_circular(T1, Y1, i_ca)
        events.append({"name": "mars_capture_burn", "t": float(t_ca), "i": int(i_ca), "dv": float(dv_cap)})

        # Start post-capture right at CA state, with dv applied (HELIocentric state!)
        y_cap0 = np.array(Y1[int(i_ca)], dtype=float)
        y_cap0[2:4] += np.asarray(dv_cap_vec, dtype=float)

        # Continue sim in heliocentric dynamics so Mars keeps orbiting the Sun
        t_end2 = float(t_ca) + float(post_capture_time)

        # Keep dt same (or slightly smaller if you want smoother Mars orbit)
        T2, Y2 = integrate_fn(derivs_fn, y_cap0, float(t_ca), float(t_end2), dt=float(dt), method=method)

        # Trim transfer to CA so no snapback when concatenating
        T1 = T1[: int(i_ca) + 1]
        Y1 = Y1[: int(i_ca) + 1]

    return T1, Y1, T2, Y2, events, metrics
