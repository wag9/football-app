"""Append-only prediction log helpers (standard library only, so they are easy to test).

Rules enforced here
  1. A prediction is recorded only if now <= kickoff - MIN_LEAD.
  2. Every new row carries kickoff_utc and predicted_at_utc.
  3. Existing rows are never rewritten. New rows are appended after them.
  4. A fixture (Date, HomeTeam, AwayTeam) is logged at most once.

Assumption to verify: football-data's fixtures.csv "Time" column is UK time
(FIXTURE_TZ). If it were local time of the stadium instead, the computed kickoff
could be up to 1-2 hours off for continental leagues. MIN_LEAD (2 hours) is a
safety margin that covers that uncertainty.
"""
from __future__ import annotations

import csv
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    raise SystemExit("Python 3.9+ is required (zoneinfo).")

FIXTURE_TZ = "Europe/London"
MIN_LEAD = timedelta(hours=2)

LEGACY_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "pH", "pD", "pA", "model", "logged_on"]
NEW_COLUMNS = ["kickoff_utc", "predicted_at_utc"]
COLUMNS = LEGACY_COLUMNS + NEW_COLUMNS

_DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d")
_TIME_FORMATS = ("%H:%M", "%H:%M:%S")


class SchemaError(Exception):
    """The log file does not have the columns we expect."""


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def _blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return str(x).strip().lower() in ("", "nan", "nat", "none")


def parse_date(text) -> str | None:
    """Return an ISO date string (YYYY-MM-DD) or None if the text is not a valid date."""
    if _blank(text):
        return None
    token = str(text).strip().split(" ")[0]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(token, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_time(text) -> time | None:
    if _blank(text):
        return None
    token = str(text).strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(token, fmt).time()
        except ValueError:
            continue
    return None


def _zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        raise SystemExit(
            f"Time zone data for '{tz_name}' not found. On Windows run: pip install tzdata"
        )


def kickoff_utc(date_text, time_text, tz_name: str = FIXTURE_TZ):
    """Return (kickoff as timezone-aware UTC datetime or None, status).

    status is one of: ok, no_time, bad_time, bad_date.
    If the time is missing or malformed we use 00:00 on the match date. That is
    deliberately conservative: it can only make us stricter, never let a late
    prediction through.
    """
    iso = parse_date(date_text)
    if iso is None:
        return None, "bad_date"
    y, m, d = (int(p) for p in iso.split("-"))
    t = parse_time(time_text)
    if t is None:
        status = "no_time" if _blank(time_text) else "bad_time"
        t = time(0, 0)
    else:
        status = "ok"
    local = datetime(y, m, d, t.hour, t.minute, tzinfo=_zone(tz_name))
    return local.astimezone(timezone.utc), status


# ----------------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------------
@dataclass
class Decision:
    ok: bool
    reason: str  # ok | bad_date | too_late
    date: str | None
    kickoff: datetime | None
    time_status: str


def can_predict(date_text, time_text, now_utc: datetime,
                tz_name: str = FIXTURE_TZ, min_lead: timedelta = MIN_LEAD) -> Decision:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    ko, status = kickoff_utc(date_text, time_text, tz_name)
    if ko is None:
        return Decision(False, "bad_date", None, None, status)
    iso = parse_date(date_text)
    if now_utc > ko - min_lead:
        return Decision(False, "too_late", iso, ko, status)
    return Decision(True, "ok", iso, ko, status)


def check_probs(pH, pD, pA, tol: float = 1e-3) -> str | None:
    """Return None if the three probabilities are valid, otherwise a reason string."""
    vals = []
    for name, v in (("pH", pH), ("pD", pD), ("pA", pA)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return f"{name} is not a number"
        if not math.isfinite(f):
            return f"{name} is not finite"
        if f < 0 or f > 1:
            return f"{name} is outside [0, 1]"
        vals.append(f)
    if abs(sum(vals) - 1.0) > tol:
        return f"probabilities sum to {sum(vals):.4f}"
    return None


def timing_status(row: dict) -> str:
    """Classify a log row: verified_pre_kickoff | violation | legacy_unverified | invalid_timestamp."""
    p = (row.get("predicted_at_utc") or "").strip()
    k = (row.get("kickoff_utc") or "").strip()
    if not p or not k:
        return "legacy_unverified"
    try:
        p_dt, k_dt = datetime.fromisoformat(p), datetime.fromisoformat(k)
    except ValueError:
        return "invalid_timestamp"
    if p_dt.tzinfo is None or k_dt.tzinfo is None:
        return "invalid_timestamp"
    return "verified_pre_kickoff" if p_dt < k_dt else "violation"


# ----------------------------------------------------------------------------
# Planning (pure: no file access)
# ----------------------------------------------------------------------------
@dataclass
class Plan:
    new_rows: list = field(default_factory=list)
    skipped: list = field(default_factory=list)   # (label, reason)
    warnings: list = field(default_factory=list)


def row_key(row: dict) -> tuple:
    return (str(row["Date"])[:10], row["HomeTeam"], row["AwayTeam"])


def _clean(x) -> str:
    return "" if _blank(x) else str(x).strip()


def plan_predictions(fixtures, existing_rows, played_keys, known_teams, predict_fn,
                     now_utc: datetime, model_version: str, tz_name: str = FIXTURE_TZ,
                     min_lead: timedelta = MIN_LEAD, local_date: str | None = None) -> Plan:
    """Decide which fixtures get a new prediction row. Does not touch any file."""
    plan = Plan()
    existing_keys = {row_key(r) for r in existing_rows}
    seen = set()
    local_date = local_date or datetime.now().date().isoformat()
    predicted_at = now_utc.astimezone(timezone.utc).isoformat(timespec="seconds")

    for fx in fixtures:
        home, away = _clean(fx.get("HomeTeam")), _clean(fx.get("AwayTeam"))
        label = f"{home or '?'} vs {away or '?'} ({_clean(fx.get('Date')) or '?'})"

        if not home or not away or home == away:
            plan.skipped.append((label, "malformed"))
            continue
        dec = can_predict(fx.get("Date"), fx.get("Time"), now_utc, tz_name, min_lead)
        if dec.reason == "bad_date":
            plan.skipped.append((label, "bad_date"))
            continue

        key = (dec.date, home, away)
        if key in played_keys:
            plan.skipped.append((label, "already_played"))
            continue
        if key in seen:
            plan.skipped.append((label, "duplicate_in_fixtures"))
            continue
        seen.add(key)
        if key in existing_keys:
            plan.skipped.append((label, "already_logged"))
            continue
        if not dec.ok:
            plan.skipped.append((label, "too_late"))
            continue
        if home not in known_teams or away not in known_teams:
            plan.skipped.append((label, "unknown_team"))
            continue

        try:
            probs = predict_fn(home, away)
            pH, pD, pA = round(float(probs["H"]), 4), round(float(probs["D"]), 4), round(float(probs["A"]), 4)
        except (ValueError, KeyError, TypeError) as e:
            plan.skipped.append((label, f"model_error: {e}"))
            continue
        err = check_probs(pH, pD, pA)
        if err:
            plan.skipped.append((label, f"bad_probs: {err}"))
            continue

        for r in existing_rows:
            if (r["HomeTeam"], r["AwayTeam"]) == (home, away) and str(r["Date"])[:10] != dec.date \
                    and (str(r["Date"])[:10], home, away) not in played_keys:
                plan.warnings.append(
                    f"possible reschedule: {home} vs {away} was logged for {str(r['Date'])[:10]} "
                    f"and has no result; new row for {dec.date}. The old row stays as it is."
                )
                break
        if dec.time_status != "ok":
            plan.warnings.append(f"{label}: kickoff time {dec.time_status}, assumed 00:00 (strict).")

        plan.new_rows.append({
            "Date": dec.date, "HomeTeam": home, "AwayTeam": away,
            "pH": f"{pH:.4f}", "pD": f"{pD:.4f}", "pA": f"{pA:.4f}",
            "model": model_version, "logged_on": local_date,
            "kickoff_utc": dec.kickoff.isoformat(timespec="seconds"),
            "predicted_at_utc": predicted_at,
        })
    return plan


# ----------------------------------------------------------------------------
# File access
# ----------------------------------------------------------------------------
def read_log(path):
    """Return (header, rows) with every value as text. Missing file -> ([], [])."""
    p = Path(path)
    if not p.exists():
        return [], []
    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _detect_newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def append_rows(path, rows) -> int:
    """Append rows to the log without touching existing bytes. Returns rows written."""
    p = Path(path)
    is_new = not p.exists()
    if not is_new:
        header, _ = read_log(p)
        if header != COLUMNS:
            raise SchemaError(f"{p} has an old schema {header}. Run migrate_log.py first.")
    before = b"" if is_new else p.read_bytes()
    nl = _detect_newline(before)

    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator=nl)
        if is_new:
            w.writerow(COLUMNS)
        elif before and not before.endswith((b"\n", b"\r")):
            f.write(nl)
        for r in rows:
            w.writerow([r[c] for c in COLUMNS])

    after = p.read_bytes()
    if not after.startswith(before):
        p.write_bytes(before)
        raise RuntimeError("Append changed existing bytes; original restored.")
    return len(rows)


def migrate_log(path, backup_dir):
    """Add the two new (blank) columns to an old-schema log. Returns (status, backup_path)."""
    p = Path(path)
    if not p.exists():
        return "missing", None
    header, rows = read_log(p)
    if header == COLUMNS:
        return "already_current", None
    if header != LEGACY_COLUMNS:
        raise SchemaError(f"Unexpected columns in {p}: {header}")

    raw = p.read_bytes()
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"{p.stem}.{stamp}{p.suffix}"
    shutil.copy2(p, backup)

    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator=_detect_newline(raw))
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r[c] for c in LEGACY_COLUMNS] + ["", ""])

    h2, rows2 = read_log(tmp)
    same = (h2 == COLUMNS and len(rows2) == len(rows)
            and all(a[c] == b[c] for a, b in zip(rows, rows2) for c in LEGACY_COLUMNS))
    if not same:
        tmp.unlink()
        raise SchemaError("Verification failed; the original file was left untouched.")
    os.replace(tmp, p)
    return "migrated", backup