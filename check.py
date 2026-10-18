# check.py -- python check.py [E0|SP1|D1|I1|F1] [--matches] [--today YYYY-MM-DD]

import sys
from datetime import date
from pathlib import Path

import evaluation as ev
import prediction_log as pl
from leagues import LEAGUES

# Model labels that you have CONFIRMED are the same model can be merged here, for example
#   {"elo_only K30 HA75 NEW1400 hist0304": "elo_only K30 HA75 NEW1400"}
# Left empty on purpose: different labels are treated as different models.
MODEL_ALIASES = {}


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def signed(x):
    return "n/a" if x is None else f"{x:+.4f}"


def describe(s):
    """Lines describing one summary (one model, one stratum)."""
    lines = []
    waiting = s["upcoming"] + s["awaiting"] + s["overdue"]
    lines.append(f"predictions: logged {s['logged']} | eligible {s['eligible']} | completed {s['completed']}"
                 f" | waiting for result {waiting} (upcoming {s['upcoming']}, awaiting {s['awaiting']},"
                 f" overdue {s['overdue']})")
    excluded = []
    if s["invalid_probs"]:
        excluded.append(f"invalid probabilities {s['invalid_probs']}")
    if s["result_conflict"]:
        excluded.append(f"conflicting results {s['result_conflict']}")
    if excluded:
        lines.append("excluded from scores: " + ", ".join(excluded))
    if s["overdue"]:
        lines.append(f"note: {s['overdue']} match(es) have had no result for more than {ev.OVERDUE_DAYS} days "
                     "(probably postponed). They are not scored.")

    m = s["metrics"]
    if m:
        pct = 100 * m["hits"] / m["n"]
        lines.append(f"top-pick accuracy {m['hits']}/{m['n']} = {pct:.0f}% | log loss {f4(m['logloss'])}"
                     f" (se {f4(m['logloss_se'])}) | Brier {f4(m['brier'])}"
                     f"   [uniform guess: {ev.UNIFORM_LOGLOSS:.4f} / {ev.UNIFORM_BRIER:.4f}]")
        b = s["bookmaker"]
        if b:
            lines.append(f"bookmaker, same matches that have odds (n={b['n']}): log loss {f4(b['book_logloss'])}"
                         f" | Brier {f4(b['book_brier'])}")
            lines.append(f"model on exactly those (n={b['n']}): log loss {f4(b['model_logloss'])}"
                         f" | Brier {f4(b['model_brier'])}"
                         f" | model minus bookmaker (log loss) {signed(b['diff'])} (se {f4(b['diff_se'])},"
                         " negative = model better)")
        else:
            lines.append("bookmaker: no completed match has valid odds, so no comparison")
        if m["n"] < ev.MIN_N:
            lines.append(f"note: only {m['n']} completed matches (< {ev.MIN_N}). Too few to conclude anything.")
    elif s["scored"]:
        lines.append("no completed matches yet")
    else:
        lines.append("not scored")
    return lines


def report_league(code, cfg, today, show_matches):
    out = [f"\n{'=' * 72}", f"{code}  {cfg['name']}", "=" * 72]
    log_path, results_path = Path(cfg["log"]), Path(cfg["matches"])
    if not log_path.exists():
        return out + ["no prediction log for this league yet"]
    if not results_path.exists():
        return out + [f"results file {results_path} is missing"]

    _, rows = pl.read_log(log_path)
    outcomes, odds, conflicts = ev.load_results(results_path)
    entries, stats = ev.prepare(rows, outcomes, odds, conflicts, today, MODEL_ALIASES)
    out.append(f"log rows: {stats['rows']} | duplicate fixtures dropped (first kept): "
               f"{stats['duplicates_dropped']} | unreadable rows: {stats['unreadable_rows']}")
    if not entries:
        return out + ["nothing to evaluate"]

    for model, by_stratum, combined in ev.evaluate(entries):
        out.append(f"\nmodel: {model}")
        for stratum, summary in by_stratum:
            out.append(f"  [{ev.STRATUM_TEXT[stratum]}]")
            out += ["    " + line for line in describe(summary)]
        if combined["metrics"] and len([s for s, _ in by_stratum if s in ev.SCORED_STRATA]) > 1:
            m, b = combined["metrics"], combined["bookmaker"]
            extra = f" | bookmaker {f4(b['book_logloss'])} on {b['n']}" if b else ""
            out.append(f"  [reconciliation only, timing strata mixed] completed {m['n']} | top picks {m['hits']}"
                       f" | log loss {f4(m['logloss'])}{extra}")

    if show_matches:
        out.append("\ncompleted matches (stratum | match | pH pD pA | result)")
        for e in sorted((e for e in entries if e["status"] == "completed"), key=lambda e: e["key"]):
            p = e["probs"]
            out.append(f"  {e['stratum']:21s} {e['key'][0]} {e['key'][1]} v {e['key'][2]} | "
                       f"{p['H']:.3f} {p['D']:.3f} {p['A']:.3f} | {e['outcome']}")
    return out


def main(argv):
    show_matches = "--matches" in argv
    today = date.today()
    if "--today" in argv:
        today = date.fromisoformat(argv[argv.index("--today") + 1])
    args = [a for i, a in enumerate(argv) if not a.startswith("--")
            and not (i > 0 and argv[i - 1] == "--today")]
    codes = args or list(LEAGUES)
    for c in codes:
        if c not in LEAGUES:
            sys.exit(f"Unknown league '{c}'. Choose one of: {', '.join(LEAGUES)}")

    print(f"evaluation date: {today} | model labels are never merged unless listed in MODEL_ALIASES")
    print("scores on few matches are noisy; the first thing to read is 'completed', not the score")
    for code in codes:
        print("\n".join(report_league(code, LEAGUES[code], today, show_matches)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))