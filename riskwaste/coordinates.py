"""The cost of a replacement rule in risk--waste coordinates.

Any rule that replaces unit ``u`` at ``min(T_u, tau_u)`` pays the preventive cost ``c_p``
when ``tau_u < T_u`` and the failure cost ``c_f`` otherwise. Its long-run cost rate ``g``
is fixed by two numbers (Lemma 1 of the paper):

* ``P_f``   -- the probability that the rule is late, ``Pr(T <= tau)``;
* ``omega`` -- the fraction of life it discards when early, ``E[(T - tau)^+] / E[T]``;

through the identity ``Tbar g / c_p = (1 + kappa P_f) / (1 - omega)`` with
``kappa = (c_f - c_p) / c_p``. Nothing is assumed about the lifetime law, the predictor,
or how ``tau`` was chosen.
"""
import numpy as np


def kappa_from_costs(c_f, c_p):
    """Cost ratio ``kappa = (c_f - c_p) / c_p`` from the failure and preventive costs."""
    if c_p <= 0:
        raise ValueError("the preventive cost must be positive")
    return (c_f - c_p) / c_p


def coordinates(lifetimes, replacement_times):
    """Risk--waste coordinates ``(omega, P_f)`` of a rule on a fleet.

    Parameters
    ----------
    lifetimes : array of shape (n,)
        Failure time ``T_u`` of each unit.
    replacement_times : array of shape (n,)
        Planned replacement time ``tau_u`` of each unit; ``np.inf`` for a unit that is
        never replaced preventively. A unit with ``tau_u >= T_u`` counts as a failure.

    Returns
    -------
    omega, pf : float
        Wasted-life fraction and failure probability, both averaged over the same units.
        Repeat a unit in the arrays to give it more weight, as a bootstrap resample does.
    """
    T = np.asarray(lifetimes, float)
    tau = np.asarray(replacement_times, float)
    if T.shape != tau.shape or T.ndim != 1 or len(T) == 0:
        raise ValueError("lifetimes and replacement_times must be 1-D arrays of one length")
    if np.any(~np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("lifetimes must be finite and positive")
    cycle = np.minimum(T, tau)
    omega = 1.0 - cycle.mean() / T.mean()
    pf = float(np.mean(T <= tau))
    return float(omega), pf


def cost_rate(omega, pf, kappa):
    """Normalised long-run cost rate ``L = Tbar g / c_p = (1 + kappa P_f) / (1 - omega)``.

    Works elementwise on arrays. Two rules scored on the same units share ``Tbar`` and
    ``c_p``, so the ratio of their ``L`` is the ratio of their cost rates.
    """
    omega = np.asarray(omega, float)
    pf = np.asarray(pf, float)
    out = (1.0 + kappa * pf) / (1.0 - omega)
    return float(out) if out.ndim == 0 else out


def coordinates_from_costs(L1, L2, kappa1, kappa2):
    """Recover ``(omega, P_f)`` from the cost rate observed at two cost ratios.

    Theorem 1(iv): an operator who sees the normalised cost rate of one operating point
    under two price regimes recovers both coordinates exactly, without measuring wasted
    life or counting failures.
    """
    if kappa1 == kappa2:
        raise ValueError("the two cost ratios must differ")
    one_minus_omega = (kappa1 - kappa2) / (kappa1 * L2 - kappa2 * L1)
    pf = (L1 - L2) * one_minus_omega / (kappa1 - kappa2)
    return float(1.0 - one_minus_omega), float(pf)
