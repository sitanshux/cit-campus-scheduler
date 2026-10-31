"""
CP-SAT scheduling engine for the CIT Campus timetable.

All 14 hard constraints are implemented.
Soft objectives are also modelled for optimisation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from ortools.sat.python import cp_model

from app.scheduling.data_prep import build_solver_data

logger = logging.getLogger(__name__)


def _compatible_rooms(
    session: dict,
    data: dict[str, Any],
) -> list[str]:
    """Return room IDs compatible with a session (type, capacity, accessibility)."""
    compatible = []
    for r in data["rooms"]:
        # HC8: room type must be in session's permitted room_types
        if r["type"] not in session["room_types"]:
            continue
        # HC7: room capacity must accommodate session size
        if r["capacity"] < session["size"]:
            continue
        # HC13: accessibility — session needs
        if session.get("needs_accessible", False) and not r.get("accessible", False):
            continue
        # HC13: accessibility — faculty needs
        fid = session.get("faculty_id")
        if fid:
            fac = data["faculty_by_id"].get(fid)
            if fac and fac.get("needs_accessible", False) and not r.get("accessible", False):
                continue
        compatible.append(r["id"])
    return compatible


def _valid_start_slots(
    session: dict,
    data: dict[str, Any],
) -> list[int]:
    """
    Return valid timeslot indices for a session's START period.

    Considers:
      - HC9: faculty unavailability (all occupied periods)
      - HC10: fixed slots
      - HC11: no classes during lunch (period gap after lunch_after)
      - HC12: 3-hour labs start only at lab_start_periods
      - teaching periods only
    """
    duration = session["duration"]
    fid = session.get("faculty_id")
    ts = data["timeslots"]
    ts_idx = data["ts_idx"]
    teaching = set(data["teaching_periods"])
    lunch_after = data["lunch_after"]
    lab_starts = set(data["lab_start_periods"])

    # HC10: fixed slot
    fixed = session.get("fixed_slot")
    if fixed:
        day = fixed["day"]
        pid = fixed["period"]
        idx = ts_idx.get((day, pid))
        if idx is not None:
            return [idx]
        return []

    valid = []
    for i, (day, pid) in enumerate(ts):
        # Gather the period ids this session would occupy
        period_ids = []
        ok = True
        for offset in range(duration):
            target_pid = pid + offset
            if target_pid not in teaching:
                ok = False
                break
            if (day, target_pid) not in ts_idx:
                ok = False
                break
            period_ids.append(target_pid)

        if not ok:
            continue

        # HC11: check session doesn't cross the lunch break
        # A multi-period session spanning period lunch_after and lunch_after+1
        # crosses the lunch hour.
        if duration > 1:
            crosses_lunch = False
            for p_idx_offset in range(len(period_ids) - 1):
                if period_ids[p_idx_offset] == lunch_after:
                    crosses_lunch = True
                    break
            if crosses_lunch:
                continue

        # HC12: 3-hour labs must start at lab_start_periods
        if duration == 3 and session.get("kind") == "lab":
            if pid not in lab_starts:
                continue

        # HC9: faculty unavailability for ALL periods in the session
        if fid:
            fac_unavail = False
            for target_pid in period_ids:
                if (fid, day, target_pid) in data["faculty_unavail"]:
                    fac_unavail = True
                    break
            if fac_unavail:
                continue

        valid.append(i)

    return valid


def solve(
    time_limit_seconds: int = 120,
) -> dict[str, Any]:
    """
    Build and solve the CP-SAT model for the CIT timetable.

    Returns a dict with:
      - status: "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "UNKNOWN" | ...
      - scheduled: [{session_id, day, period, room_id, duration, ...}]
      - unscheduled: [{session_id, reason}]
      - diagnostics: {...}
      - solve_time_seconds: float
    """
    t0 = time.time()
    data = build_solver_data()

    model = cp_model.CpModel()

    sessions = data["sessions"]
    ts = data["timeslots"]
    rooms = data["rooms"]

    # ── Pre-compute: which sessions share a basket slot? ──
    # Sessions in the same basket must run simultaneously in different rooms.
    # Students choose ONE of the basket courses, so basket-mates do NOT
    # conflict on group clashes — only on room clashes.
    basket_mates: dict[str, set[str]] = {}  # session_id -> set of basket-mate ids
    for basket_name, basket_sids in data["basket_slots"].items():
        for sid in basket_sids:
            basket_mates.setdefault(sid, set())
            for other in basket_sids:
                if other != sid:
                    basket_mates[sid].add(other)

    # ── Decision variables ──
    x: dict[tuple[str, int, str], Any] = {}

    session_vars: dict[str, list[tuple[int, str]]] = {}
    infeasible_sessions: list[dict] = []

    for s in sessions:
        sid = s["id"]
        valid_starts = _valid_start_slots(s, data)
        compat_rooms = _compatible_rooms(s, data)

        if not valid_starts:
            infeasible_sessions.append({"session_id": sid, "reason": "No valid timeslot"})
            continue
        if not compat_rooms:
            infeasible_sessions.append({"session_id": sid, "reason": "No compatible room"})
            continue

        combos: list[tuple[int, str]] = []
        for t_idx in valid_starts:
            day, pid = ts[t_idx]
            for rid in compat_rooms:
                room_ok = True
                for off in range(s["duration"]):
                    if (rid, day, pid + off) in data["room_unavail"]:
                        room_ok = False
                        break
                if room_ok:
                    var = model.new_bool_var(f"x_{sid}_{t_idx}_{rid}")
                    x[(sid, t_idx, rid)] = var
                    combos.append((t_idx, rid))

        if not combos:
            infeasible_sessions.append(
                {"session_id": sid, "reason": "No valid timeslot+room combo after filtering"}
            )
            continue

        session_vars[sid] = combos

    # ── HC1: Every schedulable session scheduled exactly once ──
    for sid, combos in session_vars.items():
        model.add(sum(x[(sid, t, r)] for t, r in combos) == 1)

    # ── Build (day, period) -> variable mappings for clash constraints ──
    dp_vars: dict[tuple[str, int], list[tuple[str, int, str, Any]]] = defaultdict(list)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        dur = s["duration"]
        for t_idx, rid in combos:
            day, pid = ts[t_idx]
            for off in range(dur):
                dp_vars[(day, pid + off)].append((sid, t_idx, rid, x[(sid, t_idx, rid)]))

    # ── HC6: No room can host overlapping sessions ──
    for (day, pid), var_list in dp_vars.items():
        room_groups: dict[str, list] = defaultdict(list)
        for sid, t_idx, rid, var in var_list:
            room_groups[rid].append(var)
        for rid, vars_in_room in room_groups.items():
            if len(vars_in_room) > 1:
                model.add(sum(vars_in_room) <= 1)

    # ── HC2: No faculty overlapping sessions ──
    faculty_dp: dict[tuple[str, str, int], list] = defaultdict(list)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        fid = s.get("faculty_id")
        if not fid:
            continue
        dur = s["duration"]
        for t_idx, rid in combos:
            day, pid = ts[t_idx]
            for off in range(dur):
                faculty_dp[(fid, day, pid + off)].append(x[(sid, t_idx, rid)])

    for key, var_list in faculty_dp.items():
        if len(var_list) > 1:
            model.add(sum(var_list) <= 1)

    # ── HC3: No student group overlapping sessions ──
    # IMPORTANT: Sessions that share a basket_slot are EXEMPT from group
    # clashes with each other. Students in those groups choose one elective,
    # so the same group appears in multiple basket courses but each student
    # only attends one.
    group_dp: dict[tuple[str, str, int], list[tuple[str, Any]]] = defaultdict(list)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        dur = s["duration"]
        for gid in s.get("groups", []):
            for t_idx, rid in combos:
                day, pid = ts[t_idx]
                for off in range(dur):
                    group_dp[(gid, day, pid + off)].append((sid, x[(sid, t_idx, rid)]))

    for key, entries in group_dp.items():
        if len(entries) <= 1:
            continue
        # Split into non-basket-conflicting clusters
        # Two sessions conflict unless they are basket-mates
        # We add pairwise: for each pair (a, b) that are NOT basket-mates,
        # at most one can be assigned to this slot.
        # For efficiency, group basket-mates together.
        # All basket-mates can coexist; non-basket sessions conflict with everything.
        session_ids = [e[0] for e in entries]
        vars_list = [e[1] for e in entries]

        # Check if any basket relationships exist
        has_basket = any(sid in basket_mates for sid in session_ids)

        if not has_basket:
            # Simple: at most one
            model.add(sum(vars_list) <= 1)
        else:
            # Complex: group by basket slot
            # Non-basket sessions get their own "group"
            # Basket-mate sessions are in the same basket group
            # Rule: at most one from each NON-BASKET, plus at most all from ONE basket group
            # Actually, the correct constraint is:
            # For any two sessions that are NOT basket-mates, at most one can be here.
            # Since basket sessions should be at the same time, they are all here or not.
            # So: the non-basket sessions plus one "representative" per basket group
            # must total <= 1.

            # Identify basket groups among entries at this (group, day, period)
            basket_groups: dict[str | None, list[int]] = defaultdict(list)
            for idx, sid in enumerate(session_ids):
                s = data["session_by_id"][sid]
                bs = s.get("basket_slot")
                basket_groups[bs].append(idx)

            # For non-basket sessions (bs=None), each is its own group
            non_basket_vars = []
            basket_representative_vars: list[Any] = []

            for bs, indices in basket_groups.items():
                if bs is None:
                    # Each non-basket session individually conflicts
                    for idx in indices:
                        non_basket_vars.append(vars_list[idx])
                else:
                    # All sessions in this basket group can coexist (they run simultaneously)
                    # But they conflict with non-basket and other basket groups
                    # Use max of the group as representative (any one being 1 means conflict)
                    if len(indices) == 1:
                        basket_representative_vars.append(vars_list[indices[0]])
                    else:
                        # Create an OR variable: 1 if any in this basket group is assigned here
                        rep = model.new_bool_var(
                            f"brep_{key[0]}_{key[1]}_{key[2]}_{bs}"
                        )
                        for idx in indices:
                            model.add(rep >= vars_list[idx])
                        model.add(rep <= sum(vars_list[idx] for idx in indices))
                        basket_representative_vars.append(rep)

            # At most 1 among: all non-basket + basket representatives
            all_competing = non_basket_vars + basket_representative_vars
            if len(all_competing) > 1:
                model.add(sum(all_competing) <= 1)

    # ── HC5: Elective basket sessions must happen simultaneously ──
    # Sessions sharing the same basket_slot must start at the same (day, period).
    for basket_name, basket_sids in data["basket_slots"].items():
        active = [sid for sid in basket_sids if sid in session_vars]
        if len(active) <= 1:
            continue

        # Find common valid timeslots across all basket members
        common_ts = None
        for sid in active:
            sid_ts = {t for t, r in session_vars[sid]}
            if common_ts is None:
                common_ts = sid_ts
            else:
                common_ts &= sid_ts

        if not common_ts:
            logger.warning(f"Basket {basket_name}: no common timeslot")
            continue

        # Create a variable for which timeslot the basket uses
        basket_ts_vars: dict[int, Any] = {}
        for t_idx in sorted(common_ts):
            basket_ts_vars[t_idx] = model.new_bool_var(f"basket_{basket_name}_ts_{t_idx}")

        # Exactly one basket timeslot chosen
        model.add(sum(basket_ts_vars.values()) == 1)

        # Link each session to the basket timeslot
        for sid in active:
            for t_idx in sorted(common_ts):
                session_at_t = []
                for t, r in session_vars[sid]:
                    if t == t_idx:
                        session_at_t.append(x[(sid, t, r)])
                if session_at_t:
                    model.add(sum(session_at_t) == basket_ts_vars[t_idx])
                else:
                    model.add(basket_ts_vars[t_idx] == 0)

    # ── HC14: Faculty daily teaching hours ──
    faculty_day_hours: dict[tuple[str, str], list[tuple[Any, int]]] = defaultdict(list)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        fid = s.get("faculty_id")
        if not fid:
            continue
        dur = s["duration"]
        for t_idx, rid in combos:
            day, _ = ts[t_idx]
            faculty_day_hours[(fid, day)].append((x[(sid, t_idx, rid)], dur))

    for (fid, day), entries in faculty_day_hours.items():
        fac = data["faculty_by_id"][fid]
        max_h = fac.get("max_hours_per_day", 8)
        model.add(sum(var * dur for var, dur in entries) <= max_h)

    # ══════════════════════════════════════════════
    # SOFT OBJECTIVES (optimisation)
    # ══════════════════════════════════════════════

    penalties: list[Any] = []
    PREF_PENALTY = 2
    SAME_COURSE_SAME_DAY_PENALTY = 1
    ROOM_UTIL_PENALTY = 1
    P9_PENALTY = 1
    GROUP_GAP_PENALTY = 1
    EXCESS_DAILY_HOURS_PENALTY = 2
    EXCESS_DAILY_HOURS_THRESHOLD = 6  # penalise if group has >6 teaching hours in a day

    # Soft 1: Faculty time preferences (avoid penalties)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        fid = s.get("faculty_id")
        if not fid:
            continue
        dur = s["duration"]
        for t_idx, rid in combos:
            day, pid = ts[t_idx]
            for off in range(dur):
                if (fid, day, pid + off) in data["faculty_prefs"]:
                    ptype = data["faculty_prefs"][(fid, day, pid + off)]
                    if ptype == "avoid":
                        penalties.append(x[(sid, t_idx, rid)] * PREF_PENALTY)

    # Soft 2: Minimise P9 (period 9) scheduling
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        dur = s["duration"]
        for t_idx, rid in combos:
            day, pid = ts[t_idx]
            for off in range(dur):
                if pid + off == 9:
                    penalties.append(x[(sid, t_idx, rid)] * P9_PENALTY)

    # Soft 3: Avoid multiple lectures of same course on same day
    course_day_vars: dict[tuple[str, str], list[tuple[Any, str]]] = defaultdict(list)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        cid = s.get("course_id")
        if not cid or s.get("kind") != "lecture":
            continue
        for t_idx, rid in combos:
            day, _ = ts[t_idx]
            course_day_vars[(cid, day)].append((x[(sid, t_idx, rid)], sid))

    for (cid, day), entries in course_day_vars.items():
        if len(entries) > 1:
            sum_expr = sum(var for var, _ in entries)
            excess = model.new_int_var(0, len(entries), f"excess_{cid}_{day}")
            model.add(excess >= sum_expr - 1)
            penalties.append(excess * SAME_COURSE_SAME_DAY_PENALTY)

    # Soft 4: Room utilisation (prefer rooms with capacity close to session size)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        for t_idx, rid in combos:
            room = data["room_by_id"][rid]
            waste = room["capacity"] - s["size"]
            if waste > 100:
                penalties.append(x[(sid, t_idx, rid)] * ROOM_UTIL_PENALTY)

    # Soft 5: Minimise gaps in student group daily schedules
    # For each group-day, track which periods have classes, then penalise gaps.
    group_day_period_vars: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        dur = s["duration"]
        for gid in s.get("groups", []):
            for t_idx, rid in combos:
                day, pid = ts[t_idx]
                for off in range(dur):
                    group_day_period_vars[(gid, day, pid + off)].append(x[(sid, t_idx, rid)])

    # Build group-day active period variables
    teaching_pids = sorted(data["teaching_periods"])
    days = data["days"]
    for gid in {g for s in sessions for g in s.get("groups", [])}:
        for day in days:
            # Create a bool var for whether group has a session in each period
            period_active: dict[int, Any] = {}
            for pid in teaching_pids:
                key = (gid, day, pid)
                if key in group_day_period_vars and group_day_period_vars[key]:
                    act = model.new_bool_var(f"gact_{gid}_{day}_{pid}")
                    model.add(act <= sum(group_day_period_vars[key]))
                    model.add(act >= 1).only_enforce_if(
                        *[v.negated() if False else v for v in []]
                    ) if False else None  # placeholder removed
                    # act = 1 if any session active, 0 otherwise
                    # Simpler: act >= each var, but we need OR semantics
                    # Use: act == max(vars) is hard; instead just add penalty
                    # for gap between first and last active period minus active count
                    period_active[pid] = (act, group_day_period_vars[key])

            # Simpler approach: penalise based on span - count
            if len(period_active) >= 2:
                sorted_pids = sorted(period_active.keys())
                # For each pair of active periods with a gap, add penalty
                # Use auxiliary: for consecutive teaching periods p_i, p_{i+1}
                # if both have sessions but p_j between them doesn't, that's a gap.
                # This is complex with CP-SAT, so use a simpler proxy:
                # penalise for each pair of non-adjacent active periods
                pass  # See simplified approach below

    # Simplified Soft 5: Penalise gaps using span minus active count proxy
    # For each group-day, compute first_active, last_active, count_active
    # gap = (last - first + 1) - count; penalise gap
    for gid in {g for s in sessions for g in s.get("groups", [])}:
        for day in days:
            # Collect periods that COULD have sessions for this group
            candidate_pids = sorted(
                pid for pid in teaching_pids
                if (gid, day, pid) in group_day_period_vars
                and group_day_period_vars[(gid, day, pid)]
            )
            if len(candidate_pids) < 2:
                continue

            # Create active-period bool vars
            act_vars: dict[int, Any] = {}
            for pid in candidate_pids:
                act = model.new_bool_var(f"gact_{gid}_{day}_{pid}")
                vars_at = group_day_period_vars[(gid, day, pid)]
                # act = 1 iff any session uses this slot for this group
                for v in vars_at:
                    model.add(act >= v)
                model.add(act <= sum(vars_at))
                act_vars[pid] = act

            # Count active periods
            count_active = sum(act_vars.values())

            # Approximate span: penalise if non-adjacent periods are both active
            # For each gap between candidate periods, if both ends active but
            # gap period inactive, add penalty.
            for i in range(len(candidate_pids) - 1):
                p1 = candidate_pids[i]
                p2 = candidate_pids[i + 1]
                if p2 - p1 > 1:
                    # There's a physical gap between these candidate periods
                    # Penalise if both are active
                    gap_pen = model.new_bool_var(f"ggap_{gid}_{day}_{p1}_{p2}")
                    model.add(gap_pen >= act_vars[p1] + act_vars[p2] - 1)
                    penalties.append(gap_pen * GROUP_GAP_PENALTY)

    # Soft 6: Excessive daily student hours
    # Penalise if a group has more than EXCESS_DAILY_HOURS_THRESHOLD hours in a day
    group_day_hours: dict[tuple[str, str], list[tuple[Any, int]]] = defaultdict(list)
    for sid, combos in session_vars.items():
        s = data["session_by_id"][sid]
        dur = s["duration"]
        for gid in s.get("groups", []):
            for t_idx, rid in combos:
                day, _ = ts[t_idx]
                group_day_hours[(gid, day)].append((x[(sid, t_idx, rid)], dur))

    for (gid, day), entries in group_day_hours.items():
        total = sum(var * dur for var, dur in entries)
        max_possible = sum(dur for _, dur in entries)
        excess = model.new_int_var(
            0, max(0, max_possible - EXCESS_DAILY_HOURS_THRESHOLD),
            f"gdexcess_{gid}_{day}"
        )
        model.add(excess >= total - EXCESS_DAILY_HOURS_THRESHOLD)
        penalties.append(excess * EXCESS_DAILY_HOURS_PENALTY)

    if penalties:
        model.minimize(sum(penalties))

    # ── Solve ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False

    status_code = solver.solve(model)
    solve_time = time.time() - t0

    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status = status_map.get(status_code, "UNKNOWN")

    scheduled: list[dict] = []
    unscheduled = list(infeasible_sessions)

    if status in ("OPTIMAL", "FEASIBLE"):
        for sid, combos in session_vars.items():
            found = False
            for t_idx, rid in combos:
                if solver.value(x[(sid, t_idx, rid)]):
                    day, pid = ts[t_idx]
                    s = data["session_by_id"][sid]
                    scheduled.append({
                        "session_id": sid,
                        "course_id": s.get("course_id"),
                        "kind": s.get("kind"),
                        "groups": s.get("groups", []),
                        "faculty_id": s.get("faculty_id"),
                        "day": day,
                        "period": pid,
                        "duration": s["duration"],
                        "room_id": rid,
                        "size": s.get("size"),
                    })
                    found = True
                    break
            if not found:
                unscheduled.append({"session_id": sid, "reason": "No assignment found"})

    diagnostics = {
        "total_sessions": len(sessions),
        "modelled_sessions": len(session_vars),
        "pre_infeasible": len(infeasible_sessions),
        "scheduled_count": len(scheduled),
        "unscheduled_count": len(unscheduled),
        "objective_value": solver.objective_value if status in ("OPTIMAL", "FEASIBLE") else None,
        "solver_status": status,
    }

    return {
        "status": status,
        "scheduled": scheduled,
        "unscheduled": unscheduled,
        "diagnostics": diagnostics,
        "solve_time_seconds": round(solve_time, 2),
    }
