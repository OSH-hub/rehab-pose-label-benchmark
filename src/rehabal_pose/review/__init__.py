"""Simulated correction-loop implementation used by the manuscript.

Human-annotation service prototypes and unfinished external backends are
intentionally excluded from this review package.
"""

from __future__ import annotations

from .clinician_loop import (
    Candidate,
    ClinicianLoop,
    Decision,
    SimulatedClinician,
    RoundRecord,
)

__all__ = [
    "Candidate",
    "ClinicianLoop",
    "Decision",
    "SimulatedClinician",
    "RoundRecord",
]
