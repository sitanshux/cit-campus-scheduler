"""
Data preparation module.

Reads the official CIT dataset and builds lookup structures
for the CP-SAT solver, without modifying the original dataset.
"""

from __future__ import annotations

from typing import Any

from app.core.dataset import load_dataset


def build_solver_data() -> dict[str, Any]:
    """
    Load the dataset and return pre-processed lookup tables.

    Returns a dict with:
      - raw: the full dataset dict
      - days / day_idx: list of day strings and {day: index}
      - periods / period_idx: list of period dicts and {pid: index}
      - teaching_periods: list of period ids where teaching=True
      - lunch_after: period id after which lunch falls
      - lab_start_periods: list of valid lab start period ids
      - event_only_periods: set of event-only period ids
      - rooms / room_by_id: list and {id: room dict}
      - room_unavail: {(room_id, day, period_id): True}
      - faculty / faculty_by_id
      - faculty_unavail: {(faculty_id, day, period_id): True}
      - faculty_prefs: {(faculty_id, day, period_id): pref_type}
      - groups / group_by_id
      - courses / course_by_id
      - sessions: list of session dicts (the 189 sessions)
      - session_by_id
      - basket_slots: {basket_slot_name: [session_ids]}
      - timeslots: [(day, period_id)] flat list of all (day, period) combos
      - ts_idx: {(day, period_id): index into timeslots}
    """
    ds = load_dataset()

    days: list[str] = ds["calendar"]["days"]
    day_idx = {d: i for i, d in enumerate(days)}

    periods: list[dict] = ds["calendar"]["periods"]
    period_idx = {p["id"]: i for i, p in enumerate(periods)}

    teaching_periods = [p["id"] for p in periods if p["teaching"]]
    event_only = set(ds["calendar"].get("event_only_periods", []))
    lunch_after = ds["calendar"]["lunch"]["after_period"]
    lab_start_periods = ds["calendar"]["lab_start_periods"]

    # Build timeslots (only teaching periods for regular sessions)
    timeslots: list[tuple[str, int]] = []
    ts_idx: dict[tuple[str, int], int] = {}
    for d in days:
        for pid in teaching_periods:
            ts = (d, pid)
            ts_idx[ts] = len(timeslots)
            timeslots.append(ts)

    # Rooms
    rooms = ds["rooms"]
    room_by_id = {r["id"]: r for r in rooms}

    # Room unavailability
    room_unavail: dict[tuple[str, str, int], bool] = {}
    for ru in ds["room_unavailability"]:
        for pid in ru["periods"]:
            room_unavail[(ru["room_id"], ru["day"], pid)] = True

    # Faculty
    faculty = ds["faculty"]
    faculty_by_id = {f["id"]: f for f in faculty}

    faculty_unavail: dict[tuple[str, str, int], bool] = {}
    faculty_prefs: dict[tuple[str, str, int], str] = {}
    for f in faculty:
        for ua in f.get("unavailable", []):
            for pid in ua["periods"]:
                faculty_unavail[(f["id"], ua["day"], pid)] = True
        for pref in f.get("preferences", []):
            for day in pref.get("days", []):
                for pid in pref.get("periods", []):
                    faculty_prefs[(f["id"], day, pid)] = pref["type"]

    # Student groups
    groups = ds["student_groups"]
    group_by_id = {g["id"]: g for g in groups}

    # Courses
    courses = ds["courses"]
    course_by_id = {c["id"]: c for c in courses}

    # Sessions
    sessions = ds["sessions"]
    session_by_id = {s["id"]: s for s in sessions}

    # Basket slots: group sessions that share the same basket_slot
    basket_slots: dict[str, list[str]] = {}
    for s in sessions:
        bs = s.get("basket_slot")
        if bs:
            basket_slots.setdefault(bs, []).append(s["id"])

    return {
        "raw": ds,
        "days": days,
        "day_idx": day_idx,
        "periods": periods,
        "period_idx": period_idx,
        "teaching_periods": teaching_periods,
        "event_only_periods": event_only,
        "lunch_after": lunch_after,
        "lab_start_periods": lab_start_periods,
        "rooms": rooms,
        "room_by_id": room_by_id,
        "room_unavail": room_unavail,
        "faculty": faculty,
        "faculty_by_id": faculty_by_id,
        "faculty_unavail": faculty_unavail,
        "faculty_prefs": faculty_prefs,
        "groups": groups,
        "group_by_id": group_by_id,
        "courses": courses,
        "course_by_id": course_by_id,
        "sessions": sessions,
        "session_by_id": session_by_id,
        "basket_slots": basket_slots,
        "timeslots": timeslots,
        "ts_idx": ts_idx,
    }

