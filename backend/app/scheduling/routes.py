"""
API routes for the scheduling engine.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.scheduling.solver import solve

router = APIRouter(prefix="/api/schedule", tags=["schedule"])


@router.post("/generate")
def generate_schedule(time_limit_seconds: int = 120):
    """
    Run the CP-SAT solver and return the timetable.

    Returns solver status, scheduled sessions, unscheduled sessions,
    and diagnostics. Never reports success if hard constraints are violated.
    """
    try:
        result = solve(time_limit_seconds=time_limit_seconds)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Solver error: {exc}",
        )

    status = result["status"]

    if status == "INFEASIBLE":
        return {
            "success": False,
            "status": status,
            "message": "No feasible schedule exists with the given constraints.",
            "scheduled": [],
            "unscheduled": result["unscheduled"],
            "diagnostics": result["diagnostics"],
            "solve_time_seconds": result["solve_time_seconds"],
        }

    if status in ("OPTIMAL", "FEASIBLE"):
        all_scheduled = result["diagnostics"]["scheduled_count"] == result["diagnostics"]["total_sessions"]
        return {
            "success": all_scheduled,
            "status": status,
            "message": (
                "All sessions scheduled successfully."
                if all_scheduled
                else f"{result['diagnostics']['unscheduled_count']} sessions could not be scheduled."
            ),
            "scheduled": result["scheduled"],
            "unscheduled": result["unscheduled"],
            "diagnostics": result["diagnostics"],
            "solve_time_seconds": result["solve_time_seconds"],
        }

    # UNKNOWN or MODEL_INVALID
    return {
        "success": False,
        "status": status,
        "message": f"Solver returned status: {status}",
        "scheduled": result.get("scheduled", []),
        "unscheduled": result.get("unscheduled", []),
        "diagnostics": result.get("diagnostics", {}),
        "solve_time_seconds": result.get("solve_time_seconds"),
    }

