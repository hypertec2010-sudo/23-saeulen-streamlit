"""Pattern-bound objectives; independent of the general analysis' target.

These are conservative planning heuristics, not validated return forecasts.
No risk-multiple floor, stop tightening or target extension is performed.
"""
from modules.trade_watch import number

PLAN_POLICY = "pattern_plan_v30.21cg"


def bind_plan(candidate, history, *, objective, source, basis_date, consumed=False, observed_high=None):
    from modules.watch_setups import pivots
    out = dict(candidate, plan_policy=PLAN_POLICY, target=None, target_source=source,
               target_reference=number(objective), target_basis_date=str(basis_date))
    entry, target = number(out.get("entry")), number(objective)
    if entry is None or target is None or target <= 0:
        return out
    # Freeze opposition known at pattern confirmation. A nearby confirmed high
    # cannot be skipped just to manufacture a better RR.
    obstacles = [float(value) for _,value in pivots(history.High,high=True)
                 if entry < float(value) < target]
    out["target"] = min(obstacles) if obstacles else target
    if obstacles:
        out["target_source"] = "Näherer bestätigter Pivot-Widerstand vor dem Musterziel"
        out["target_obstacle"] = min(obstacles)
    observed = number(observed_high)
    spent = consumed or entry >= target or (observed is not None and observed >= out["target"])
    if out.get("phase") not in {"none","blocked","invalid"} and spent:
        out.update(phase="missed",reason_code="pattern_target_consumed",
                   reason="Bewegungsziel dieses Musters bereits abgearbeitet oder vor dem geplanten Entry; neue Struktur erforderlich.")
    return out


def candidate_rank(candidate):
    from modules.watch_setups import RANK
    from modules.watch_quality import assess
    report = assess(candidate)
    rr = report["rr_cap"]
    return (int(candidate.get("phase")=="ready" and report["allowed"]),
            RANK[candidate["phase"]], rr if rr is not None else float("-inf"))


def new_plan_check(levels):
    """Only new admissions need the revised plan. Existing ideas keep their rules."""
    package = (levels.get("source") or {}).get("watch_setup") if "source" in levels else levels
    return (isinstance(package,dict) and package.get("plan_policy")==PLAN_POLICY)
