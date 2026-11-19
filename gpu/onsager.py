"""Exact results for the infinite 2D square-lattice Ising model (J = k_B = 1)."""
import numpy as np
from scipy.special import ellipe, ellipk

TC = 2 / np.log(1 + np.sqrt(2))
KC = 1 / TC

# Finite-size scaling of the specific-heat peak: C_max(L) ~ (8/pi) K_c^2 ln L + const
CMAX_LOG_SLOPE = 8 / np.pi * KC**2


def _parts(T):
    K = 1 / np.asarray(T, dtype=float)
    t = np.tanh(2 * K)
    kappa = 2 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    return K, t, kappa**2  # scipy's ellipk/ellipe take the parameter m = kappa^2


def energy(T):
    """Energy per spin u(T)."""
    K, t, m = _parts(T)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = -1 / t * (1 + 2 / np.pi * (2 * t**2 - 1) * ellipk(m))
    return np.where(np.isfinite(u), u, -np.sqrt(2))  # u(Tc) = -sqrt(2)


def specific_heat(T):
    """Specific heat per spin c(T) (log-divergent at Tc)."""
    K, t, m = _parts(T)
    K1, E1 = ellipk(m), ellipe(m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (4 / np.pi) * (K / t) ** 2 * (
            K1 - E1 - (1 - t**2) * (np.pi / 2 + (2 * t**2 - 1) * K1))


def magnetisation(T):
    """Spontaneous magnetisation |m|(T), zero above Tc."""
    T = np.asarray(T, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        m = (1 - np.sinh(2 / T) ** -4) ** (1 / 8)
    return np.where(T < TC, m, 0.0)


def xi_row(T):
    """Exact correlation length of row/column averages above Tc (as in the notebook)."""
    T = np.asarray(T, dtype=float)
    return 1 / (np.log(1 / np.tanh(1 / T)) - 2 / T)
