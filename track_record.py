# track_record.py
"""Track record for the website: prospective, verified predictions only.

Pure functions (no file access, no FastAPI), so the rules can be tested with fake rows.
api.py loads the log and results files and calls build(). It uses the same evaluation
code (evaluation.py) as check.py, so the website and the terminal cannot disagree.
"""
from __future__ import annotations

from datetime import date

import evaluation as ev

HEADLINE_STRATUM = "prospective_verified"


def _r(x, nd=4):
    return None if x is None else round(x, nd)


def build(log_rows, outcomes, odds, conflicts, today: date, model: str):
    entries, _stats = ev.prepare(log_rows, outcomes, odds, conflicts, today)
    other_models = sorted({e["model"] for e in entries if e["model"] != model})
    mine = [e for e in entries if e["model"] == model and e["stratum"] == HEADLINE_STRATUM]

    s = ev.summarize(mine, scored=True)
    m, b = s["metrics"], s["bookmaker"]

    done = sorted((e for e in mine if e["status"] == "completed"),
                  key=lambda e: e["key"], reverse=True)
    matches = []
    for e in done:
        d, home, away = e["key"]
        _ll, _brier, hit = ev.per_match(e["probs"], e["outcome"])
        matches.append({
            "date": d, "home": home, "away": away,
            "prob": {"home": _r(e["probs"]["H"]), "draw": _r(e["probs"]["D"]), "away": _r(e["probs"]["A"])},
            "result": e["outcome"],
            "top_pick_correct": hit,
        })

    return {
        "model": model,
        "evaluated_on": today.isoformat(),
        "logged": s["logged"],
        "completed": s["completed"],
        "waiting": s["upcoming"] + s["awaiting"] + s["overdue"],
        "min_matches": ev.MIN_N,
        "low_sample": s["completed"] < ev.MIN_N,
        "uniform_logloss": _r(ev.UNIFORM_LOGLOSS),
        "metrics": None if m is None else {
            "n": m["n"],
            "top_pick_hits": m["hits"],
            "top_pick_accuracy": _r(m["hits"] / m["n"]),
            "logloss": _r(m["logloss"]),
            "logloss_se": _r(m["logloss_se"]),
            "brier": _r(m["brier"]),
        },
        "bookmaker": None if b is None else {
            "n": b["n"],
            "model_logloss": _r(b["model_logloss"]),
            "bookmaker_logloss": _r(b["book_logloss"]),
            "diff": _r(b["diff"]),
            "diff_se": _r(b["diff_se"]),
        },
        "matches": matches,
        "other_models": other_models,
    }