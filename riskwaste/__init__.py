"""riskwaste: risk--waste geometry for condition-based replacement.

Place any replacement rule at a point ``(omega, P_f)``, find where the optimal threshold
on a predicted remaining life lies, and state how precisely a finite fleet can resolve a
cost comparison. Companion code to Le and Nguyen, "Risk--waste geometry for
condition-based replacement: Optimality and finite-fleet estimation".
"""
from .age import age_coordinates, optimal_age
from .bootstrap import (PairedResult, ThresholdChoice, choose_threshold,
                        paired_bootstrap)
from .coordinates import (coordinates, coordinates_from_costs, cost_rate,
                          kappa_from_costs)
from .threshold import ThresholdPolicy, kappa_star, optimal_intervals, optimum

__version__ = "0.1.0"

__all__ = [
    "coordinates", "cost_rate", "coordinates_from_costs", "kappa_from_costs",
    "ThresholdPolicy", "kappa_star", "optimum", "optimal_intervals",
    "age_coordinates", "optimal_age",
    "paired_bootstrap", "PairedResult", "choose_threshold", "ThresholdChoice",
]
