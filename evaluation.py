"""Evaluation of the prediction logs. Standard library only, so it is easy to test.

Design rules
  * Scores are computed per league, per model label, per timing stratum. They are never
    pooled across model labels.
  * Only rows whose result is known are scored. Postponed or not-yet-updated matches
    stay "waiting" and are never guessed.
  * Rows that cannot have been predicted before kickoff are counted but never scored.
  * The bookmaker is compared only on matches that were scored for the model AND have
    valid odds; the model is re-scored on exactly that subset.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import date
from pathlib import Path

import prediction_log as pl

OUTCOMES = ("H", "D", "A")
EPS = 1e-15
OVERDUE_DAYS = 14                      # no result this long after the match date: likely postponed
UNIFORM_LOGLOSS = math.log(3)
UNIFORM_BRIER = 2.0 / 3.0
MIN_N = 30                             # below this a score is mostly noise

SCORED_STRATA = ("prospective_verified", "legacy_prior_day", "legacy_same_day")
EXCLUDED_STRATA = ("timing_violation", "timing_invalid")
STRATUM_ORDER = SCORED_STRATA + EXCLUDED_STRATA
STRATUM_TEXT = {
    "prospective_verified": "PROSPECTIVE, verified (predicted_at_utc is before kickoff_utc)",
    "legacy_prior_day": "LEGACY, logged on an earlier day than the match (no timestamp, relies on logged_on)",
    "legacy_same_day": "LEGACY, logged on the match day (no timestamp, cannot prove it was before kickoff)",
    "timing_violation": "INVALID, logged after kickoff. Counted, never scored",
    "timing_invalid": "INVALID, unreadable timestamp or date. Counted, never scored",
}


# ----------------------------------------------------------------------------
# Loading results
# ----------------------------------------------------------------------------
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _bookmaker(row):
    """Normalised implied probabilities from AvgH/AvgD/AvgA, or None if any odd is missing/invalid."""
    odds = [_num(row.get(c)) for c in ("AvgH", "AvgD", "AvgA")]
    if any(o is None or o <= 1.0 for o in odds):
        return None
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return {k: v / s for k, v in zip(OUTCOMES, inv)}


def load_results(path):
    """Return (outcomes, odds, conflicts). Keys are (iso_date, home, away).

    A fixture that appears twice with different results is a conflict and is never scored.
    Identical duplicates are harmless and collapse to one.
    """
    outcomes, odds, conflicts = {}, {}, set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d = pl.parse_date(r.get("Date"))
            h, a = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
            ftr = (r.get("FTR") or "").strip().upper()
            if d is None or not h or not a or ftr not in OUTCOMES:
                continue
            key = (d, h, a)
            if key in outcomes and outcomes[key] != ftr:
                conflicts.add(key)
                continue
            outcomes.setdefault(key, ftr)
            odds.setdefault(key, _bookmaker(r))
    return outcomes, odds, conflicts


# ----------------------------------------------------------------------------
# Classifying log rows
# ----------------------------------------------------------------------------
def stratum(row, iso_date):
    t = pl.timing_status(row)
    if t == "verified_pre_kickoff":
        return "prospective_verified"
    if t == "violation":
        return "timing_violation"
    if t == "invalid_timestamp":
        return "timing_invalid"
    logged = pl.parse_date(row.get("logged_on"))        # legacy row: no timestamps
    if logged is None:
        return "timing_invalid"
    if logged > iso_date:
        return "timing_violation"
    return "legacy_same_day" if logged == iso_date else "legacy_prior_day"


def prepare(log_rows, outcomes, odds, conflicts, today: date, aliases=None):
    """Turn raw log rows into entries. Returns (entries, stats). The first row for a fixture wins."""
    aliases = aliases or {}
    entries, seen = [], set()
    stats = {"rows": 0, "duplicates_dropped": 0, "unreadable_rows": 0}
    for row in log_rows:
        stats["rows"] += 1
        d = pl.parse_date(row.get("Date"))
        home, away = (row.get("HomeTeam") or "").strip(), (row.get("AwayTeam") or "").strip()
        if d is None or not home or not away:
            stats["unreadable_rows"] += 1
            continue
        key = (d, home, away)
        if key in seen:
            stats["duplicates_dropped"] += 1
            continue
        seen.add(key)

        model = (row.get("model") or "").strip() or "(no model label)"
        model = aliases.get(model, model)
        err = pl.check_probs(row.get("pH"), row.get("pD"), row.get("pA"))
        probs = None if err else {"H": float(row["pH"]), "D": float(row["pD"]), "A": float(row["pA"])}

        if err:
            status = "invalid_probs"
        elif key in conflicts:
            status = "result_conflict"
        elif key in outcomes:
            status = "completed"
        else:
            match_day = date.fromisoformat(d)
            if match_day > today:
                status = "upcoming"
            elif (today - match_day).days > OVERDUE_DAYS:
                status = "overdue"
            else:
                status = "awaiting"

        entries.append({
            "key": key, "model": model, "stratum": stratum(row, d), "probs": probs,
            "status": status, "outcome": outcomes.get(key) if status == "completed" else None,
            "book": odds.get(key) if status == "completed" else None,
        })
    return entries, stats


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def mean(v):
    return sum(v) / len(v)


def standard_error(v):
    n = len(v)
    if n < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n)


def normalise(p):
    c = {k: min(max(float(p[k]), 0.0), 1.0) for k in OUTCOMES}
    s = sum(c.values())
    return {k: c[k] / s for k in OUTCOMES}


def per_match(p, outcome):
    """(log loss, Brier, top pick correct) for one match. Brier = sum over H/D/A of (p - actual)^2."""
    p = normalise(p)
    log_loss = -math.log(max(p[outcome], EPS))
    brier = sum((p[k] - (1.0 if k == outcome else 0.0)) ** 2 for k in OUTCOMES)
    top = max(OUTCOMES, key=lambda k: p[k])          # ties resolve in H, D, A order
    return log_loss, brier, top == outcome


def summarize(entries, scored=True):
    counts = Counter(e["status"] for e in entries)
    out = {
        "logged": len(entries),
        "invalid_probs": counts["invalid_probs"],
        "result_conflict": counts["result_conflict"],
        "eligible": (len(entries) - counts["invalid_probs"]) if scored else 0,
        "completed": counts["completed"] if scored else 0,
        "upcoming": counts["upcoming"], "awaiting": counts["awaiting"], "overdue": counts["overdue"],
        "metrics": None, "bookmaker": None, "scored": scored,
    }
    done = [e for e in entries if e["status"] == "completed"] if scored else []
    if not done:
        return out

    vals = [per_match(e["probs"], e["outcome"]) for e in done]
    ll = [v[0] for v in vals]
    out["metrics"] = {
        "n": len(done), "hits": sum(v[2] for v in vals),
        "logloss": mean(ll), "logloss_se": standard_error(ll),
        "brier": mean([v[1] for v in vals]),
    }

    comparable = [e for e in done if e["book"]]
    if comparable:
        m = [per_match(e["probs"], e["outcome"]) for e in comparable]
        b = [per_match(e["book"], e["outcome"]) for e in comparable]
        diffs = [mi[0] - bi[0] for mi, bi in zip(m, b)]
        out["bookmaker"] = {
            "n": len(comparable),
            "model_logloss": mean([x[0] for x in m]), "book_logloss": mean([x[0] for x in b]),
            "model_brier": mean([x[1] for x in m]), "book_brier": mean([x[1] for x in b]),
            "diff": mean(diffs), "diff_se": standard_error(diffs),
        }
    return out


def evaluate(entries):
    """Return [(model, [(stratum, summary)], combined_scored_summary)] in order of first appearance."""
    models = {}
    for e in entries:
        models.setdefault(e["model"], []).append(e)
    result = []
    for model, rows in models.items():
        by_stratum = []
        for s in STRATUM_ORDER:
            part = [e for e in rows if e["stratum"] == s]
            if part:
                by_stratum.append((s, summarize(part, scored=s in SCORED_STRATA)))
        scored_rows = [e for e in rows if e["stratum"] in SCORED_STRATA]
        result.append((model, by_stratum, summarize(scored_rows, scored=True)))
    return result