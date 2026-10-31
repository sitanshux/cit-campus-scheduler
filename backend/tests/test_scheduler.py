"""
Comprehensive tests for the CIT Campus Scheduling Engine.

Tests cover:
  1. Solver execution
  2. Session completeness (all 189 sessions)
  3. Faculty clash detection
  4. Student group clash detection
  5. Room clash detection
  6. Room capacity and type compatibility
  7. Faculty and room unavailability
  8. Fixed slots
  9. Lab timeslot constraints
  10. Lunch break constraint
  11. Accessibility
  12. Faculty daily hours
  13. Basket (elective) simultaneous scheduling
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dataset import load_dataset
from app.scheduling.data_prep import build_solver_data
from app.scheduling.solver import solve


# ── Run the solver once and cache ──

_result: dict | None = None


def get_result() -> dict:
    global _result
    if _result is None:
        _result = solve(time_limit_seconds=120)
    return _result


@pytest.fixture(scope="module")
def result():
    return get_result()


@pytest.fixture(scope="module")
def data():
    return build_solver_data()


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def scheduled(result):
    return result["scheduled"]


# ── 1. Solver execution ──


class TestSolverExecution:
    def test_solver_returns_result(self, result):
        assert result is not None
        assert "status" in result

    def test_solver_finds_solution(self, result):
        assert result["status"] in ("OPTIMAL", "FEASIBLE"), (
            f"Solver status: {result['status']}, "
            f"unscheduled: {result.get('unscheduled', [])}"
        )

    def test_diagnostics_present(self, result):
        diag = result["diagnostics"]
        assert "total_sessions" in diag
        assert "scheduled_count" in diag


# ── 2. Session completeness ──


class TestSessionCompleteness:
    def test_total_sessions_in_dataset(self, dataset):
        """Verify the dataset has 189 sessions."""
        assert len(dataset["sessions"]) == 189

    def test_all_sessions_scheduled(self, result, dataset):
        """Every session from the dataset must be scheduled."""
        scheduled_ids = {s["session_id"] for s in result["scheduled"]}
        all_ids = {s["id"] for s in dataset["sessions"]}
        missing = all_ids - scheduled_ids
        assert len(missing) == 0, f"Missing sessions: {missing}"


# ── 3. Faculty clash detection ──


class TestFacultyClashes:
    def test_no_faculty_overlap(self, scheduled):
        """No faculty teaches two sessions at the same (day, period)."""
        faculty_slots: dict[tuple, list] = defaultdict(list)
        for s in scheduled:
            fid = s.get("faculty_id")
            if not fid:
                continue
            for off in range(s["duration"]):
                key = (fid, s["day"], s["period"] + off)
                faculty_slots[key].append(s["session_id"])

        clashes = {k: v for k, v in faculty_slots.items() if len(v) > 1}
        assert len(clashes) == 0, f"Faculty clashes: {clashes}"


# ── 4. Student group clash detection ──


class TestGroupClashes:
    def test_no_group_overlap(self, scheduled, data):
        """No student group has two sessions at the same (day, period).

        Elective basket sessions are exempt: they run simultaneously by
        design and students choose only one course from the basket.
        """
        # Build set of basket-mate pairs
        basket_mate_pairs: set[frozenset[str]] = set()
        for basket_name, sids in data["basket_slots"].items():
            for i, a in enumerate(sids):
                for b in sids[i + 1:]:
                    basket_mate_pairs.add(frozenset([a, b]))

        group_slots: dict[tuple, list] = defaultdict(list)
        for s in scheduled:
            for gid in s.get("groups", []):
                for off in range(s["duration"]):
                    key = (gid, s["day"], s["period"] + off)
                    group_slots[key].append(s["session_id"])

        clashes = {}
        for k, v in group_slots.items():
            if len(v) <= 1:
                continue
            # Filter: only flag if any pair is NOT basket-mates
            real_clashes = []
            for i, a in enumerate(v):
                for b in v[i + 1:]:
                    if frozenset([a, b]) not in basket_mate_pairs:
                        real_clashes.append((a, b))
            if real_clashes:
                clashes[k] = real_clashes

        assert len(clashes) == 0, f"Group clashes (non-basket): {clashes}"


# ── 5. Room clash detection ──


class TestRoomClashes:
    def test_no_room_overlap(self, scheduled):
        """No room hosts two sessions at the same (day, period)."""
        room_slots: dict[tuple, list] = defaultdict(list)
        for s in scheduled:
            for off in range(s["duration"]):
                key = (s["room_id"], s["day"], s["period"] + off)
                room_slots[key].append(s["session_id"])

        clashes = {k: v for k, v in room_slots.items() if len(v) > 1}
        assert len(clashes) == 0, f"Room clashes: {clashes}"


# ── 6. Room capacity and type compatibility ──


class TestRoomCompatibility:
    def test_room_capacity(self, scheduled, data):
        """Room capacity must be >= session size."""
        violations = []
        for s in scheduled:
            room = data["room_by_id"][s["room_id"]]
            if room["capacity"] < s["size"]:
                violations.append(
                    f"{s['session_id']}: size={s['size']} > room {s['room_id']} cap={room['capacity']}"
                )
        assert len(violations) == 0, f"Capacity violations: {violations}"

    def test_room_type(self, scheduled, data):
        """Room type must match session's permitted room_types."""
        violations = []
        for s in scheduled:
            room = data["room_by_id"][s["room_id"]]
            session = data["session_by_id"][s["session_id"]]
            if room["type"] not in session["room_types"]:
                violations.append(
                    f"{s['session_id']}: room {s['room_id']} type={room['type']} "
                    f"not in {session['room_types']}"
                )
        assert len(violations) == 0, f"Room type violations: {violations}"


# ── 7. Unavailability ──


class TestUnavailability:
    def test_faculty_unavailability(self, scheduled, data):
        """Sessions must not be placed when faculty is unavailable."""
        violations = []
        for s in scheduled:
            fid = s.get("faculty_id")
            if not fid:
                continue
            for off in range(s["duration"]):
                if (fid, s["day"], s["period"] + off) in data["faculty_unavail"]:
                    violations.append(
                        f"{s['session_id']}: faculty {fid} unavailable "
                        f"on {s['day']} P{s['period'] + off}"
                    )
        assert len(violations) == 0, f"Faculty unavailability violations: {violations}"

    def test_room_unavailability(self, scheduled, data):
        """Sessions must not use rooms when they are unavailable."""
        violations = []
        for s in scheduled:
            for off in range(s["duration"]):
                if (s["room_id"], s["day"], s["period"] + off) in data["room_unavail"]:
                    violations.append(
                        f"{s['session_id']}: room {s['room_id']} unavailable "
                        f"on {s['day']} P{s['period'] + off}"
                    )
        assert len(violations) == 0, f"Room unavailability violations: {violations}"


# ── 8. Fixed slots ──


class TestFixedSlots:
    def test_fixed_slots_respected(self, scheduled, data):
        """Sessions with fixed_slot must be at the specified day/period."""
        violations = []
        for s in scheduled:
            session = data["session_by_id"][s["session_id"]]
            fixed = session.get("fixed_slot")
            if fixed:
                if s["day"] != fixed["day"] or s["period"] != fixed["period"]:
                    violations.append(
                        f"{s['session_id']}: expected {fixed}, got day={s['day']} P{s['period']}"
                    )
        assert len(violations) == 0, f"Fixed slot violations: {violations}"


# ── 9. Lab constraints ──


class TestLabConstraints:
    def test_three_hour_labs_valid_start(self, scheduled, data):
        """3-hour labs must start at period 2 or 6 (lab_start_periods)."""
        lab_starts = set(data["lab_start_periods"])
        violations = []
        for s in scheduled:
            session = data["session_by_id"][s["session_id"]]
            if session.get("kind") == "lab" and session["duration"] == 3:
                if s["period"] not in lab_starts:
                    violations.append(
                        f"{s['session_id']}: lab starts at P{s['period']}, "
                        f"expected one of {lab_starts}"
                    )
        assert len(violations) == 0, f"Lab start violations: {violations}"


# ── 10. Lunch break ──


class TestLunchBreak:
    def test_no_classes_cross_lunch(self, scheduled, data):
        """No session should span periods that cross the lunch break."""
        lunch_after = data["lunch_after"]
        violations = []
        for s in scheduled:
            start = s["period"]
            end = start + s["duration"] - 1
            # If session spans from before/at lunch_after to after it
            if start <= lunch_after and end > lunch_after:
                violations.append(
                    f"{s['session_id']}: P{start}-P{end} crosses lunch after P{lunch_after}"
                )
        assert len(violations) == 0, f"Lunch violations: {violations}"


# ── 11. Accessibility ──


class TestAccessibility:
    def test_accessible_room_for_accessible_session(self, scheduled, data):
        """Sessions needing accessibility must be in accessible rooms."""
        violations = []
        for s in scheduled:
            session = data["session_by_id"][s["session_id"]]
            if session.get("needs_accessible"):
                room = data["room_by_id"][s["room_id"]]
                if not room.get("accessible"):
                    violations.append(
                        f"{s['session_id']}: needs accessible room, "
                        f"got {s['room_id']} (accessible={room.get('accessible')})"
                    )
        assert len(violations) == 0, f"Accessibility violations: {violations}"


# ── 12. Faculty daily hours ──


class TestFacultyDailyHours:
    def test_max_hours_per_day(self, scheduled, data):
        """Faculty must not exceed their max_hours_per_day."""
        faculty_day_hours: dict[tuple[str, str], int] = defaultdict(int)
        for s in scheduled:
            fid = s.get("faculty_id")
            if not fid:
                continue
            faculty_day_hours[(fid, s["day"])] += s["duration"]

        violations = []
        for (fid, day), hours in faculty_day_hours.items():
            fac = data["faculty_by_id"][fid]
            max_h = fac.get("max_hours_per_day", 8)
            if hours > max_h:
                violations.append(
                    f"{fid} on {day}: {hours}h > max {max_h}h"
                )
        assert len(violations) == 0, f"Daily hours violations: {violations}"


# ── 13. Basket (elective) simultaneous scheduling ──


class TestBasketSlots:
    def test_basket_sessions_same_timeslot(self, scheduled, data):
        """Sessions in the same basket_slot must start at the same day+period."""
        sched_map = {s["session_id"]: s for s in scheduled}
        violations = []
        for basket_name, sids in data["basket_slots"].items():
            times = set()
            for sid in sids:
                if sid in sched_map:
                    s = sched_map[sid]
                    times.add((s["day"], s["period"]))
            if len(times) > 1:
                violations.append(
                    f"Basket {basket_name}: sessions at different times: {times}"
                )
        assert len(violations) == 0, f"Basket violations: {violations}"

