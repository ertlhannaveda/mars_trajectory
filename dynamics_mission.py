import numpy as np
from constants import MU_SUN, MU_EARTH, MU_MARS, R_MARS
from bodies import mars_state

# 2D heliocentric dynamics (AU, YEAR/(2*pi), mu_sun=1).
# For stable Earth->Mars transfers with Hohmann phasing:
# - Always include Sun gravity.
# - Include Mars gravity only (smooth SOI taper) so flyby/capture can happen near encounter.
# Earth gravity is intentionally omitted (otherwise you must model Earth SOI / parking orbit properly).

SUN_SOFT_AU = 1e-6
MARS_SOFT_AU = 3e-5

R_SOI_MARS = float(R_MARS * (MU_MARS / MU_SUN) ** (2.0 / 5.0))

ENABLE_MARS_GRAVITY = True


def _soft_accel(mu: float, d: np.ndarray, soft_au: float) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    r2 = float(np.dot(d, d) + soft_au**2)
    inv = 1.0 / (r2 * np.sqrt(r2))
    return -mu * d * inv


def _soi_weight(dist: float, r_soi: float, power: int = 6) -> float:
    if r_soi <= 0.0:
        return 0.0
    x = (dist / r_soi) ** power
    return 1.0 / (1.0 + x)


def acceleration_spacecraft(t: float, r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    a = np.zeros(2, dtype=float)

    # Sun at origin
    a += _soft_accel(MU_SUN, r, SUN_SOFT_AU)

    if ENABLE_MARS_GRAVITY:
        rM, _ = mars_state(t)
        dM = r - rM
        nM = float(np.linalg.norm(dM))
        wM = _soi_weight(nM, R_SOI_MARS)
        a += wM * _soft_accel(MU_MARS, dM, MARS_SOFT_AU)

    return a


def derivs(t: float, state) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    r = state[:2]
    v = state[2:]
    a = acceleration_spacecraft(t, r)
    return np.hstack((v, a))
