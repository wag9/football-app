# data_checks.py -- read-only integrity checks for a league matches table (and fixtures).
# Pure functions: they take DataFrames and return a list of Issue tuples. They never write
# or modify anything, so they are safe to run on real data and easy to unit-test on synthetic data.
import re
from collections import namedtuple

import numpy as np
import pandas as pd

# level: "ERROR" = breaks the model/evaluation, "WARN" = suspicious, "INFO" = good to know
Issue = namedtuple("Issue", "level check n detail")

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                 "AvgH", "AvgD", "AvgA", "HS", "AS", "HST", "AST", "HC", "AC", "season"]
SEASON_RE = re.compile(r"^\d{4}$")


def _sample(df, cols, k=5):
    return df[cols].head(k).astype(str).agg(" | ".join, axis=1).tolist()


def check_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return [Issue("ERROR", "columns", len(missing), f"missing columns: {missing}")] if missing else []


def check_dates(df, today=None):
    out = []
    raw = df["Date"]
    d = pd.to_datetime(raw, errors="coerce")
    bad = d.isna()
    if bad.any():
        out.append(Issue("ERROR", "date_parse", int(bad.sum()), f"unparseable dates, e.g. {raw[bad].head(3).tolist()}"))
    if not d.dropna().is_monotonic_increasing:
        out.append(Issue("WARN", "date_order", 1, "file is not sorted by Date (data.load_matches sorts it, so only a note)"))
    if today is not None:
        fut = (d > pd.Timestamp(today)) & df["FTR"].notna()
        if fut.any():
            out.append(Issue("ERROR", "future_result", int(fut.sum()),
                             f"rows dated after {today} that already have a result: {_sample(df[fut], ['Date','HomeTeam','AwayTeam'])}"))
    return out


def check_season_labels(df):
    """Season label 'YYZZ' must match the match date: Jul 1 20YY .. Aug 31 20ZZ (loose on purpose; catches a label that is a year off)."""
    out = []
    s = df["season"].astype(str)
    badfmt = ~s.str.match(SEASON_RE)
    if badfmt.any():
        out.append(Issue("ERROR", "season_format", int(badfmt.sum()), f"bad season labels: {s[badfmt].unique()[:5].tolist()}"))
        return out
    d = pd.to_datetime(df["Date"], errors="coerce")
    y0 = 2000 + s.str[:2].astype(int)
    start = pd.to_datetime(y0.astype(str) + "-07-01")
    end = pd.to_datetime((y0 + 1).astype(str) + "-08-31")   # Aug 31, not Jun 30: the COVID-hit 2019/20 season finished in July/Aug 2020
    wrong = (d < start) | (d > end)
    if wrong.any():
        out.append(Issue("ERROR", "season_vs_date", int(wrong.sum()),
                         f"date outside its season window: {_sample(df[wrong], ['season','Date','HomeTeam','AwayTeam'])}"))
    return out


def check_duplicates(df):
    out = []
    key = ["Date", "HomeTeam", "AwayTeam"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        out.append(Issue("ERROR", "duplicate_match", int(dup.sum()), f"same Date+Home+Away more than once: {_sample(df[dup], key)}"))
    same = df["HomeTeam"] == df["AwayTeam"]
    if same.any():
        out.append(Issue("ERROR", "home_equals_away", int(same.sum()), "HomeTeam == AwayTeam"))
    # main.py and shots.py merge on (Date, Team): a team twice on one date would silently duplicate rows
    long = pd.concat([df[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}),
                      df[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})])
    twice = long.duplicated(["Date", "Team"], keep=False)
    if twice.any():
        out.append(Issue("ERROR", "team_twice_same_day", int(twice.sum()),
                         f"team plays twice on one date: {long[twice].drop_duplicates().head(5).astype(str).agg(' | '.join, axis=1).tolist()}"))
    return out


def check_results(df):
    out = []
    for c in ("FTHG", "FTAG"):
        v = pd.to_numeric(df[c], errors="coerce")
        n_nan = int(v.isna().sum())
        if n_nan:
            out.append(Issue("ERROR", f"{c}_missing", n_nan, f"{c} missing/non-numeric"))
        if (v < 0).any() or ((v.dropna() % 1) != 0).any():
            out.append(Issue("ERROR", f"{c}_invalid", int(((v < 0) | ((v % 1) != 0)).sum()), "negative or non-integer goals"))
    badftr = ~df["FTR"].isin(["H", "D", "A"])
    if badftr.any():
        out.append(Issue("ERROR", "FTR_values", int(badftr.sum()), f"FTR not in H/D/A: {df.loc[badftr, 'FTR'].unique()[:5].tolist()}"))
    hg, ag = pd.to_numeric(df["FTHG"], errors="coerce"), pd.to_numeric(df["FTAG"], errors="coerce")
    implied = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    mism = df["FTR"].isin(["H", "D", "A"]) & hg.notna() & ag.notna() & (implied != df["FTR"].values)
    if mism.any():
        out.append(Issue("ERROR", "FTR_vs_goals", int(mism.sum()),
                         f"FTR disagrees with goals: {_sample(df[mism], ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'])}"))
    return out


def check_odds(df, lo=1.0, hi=1.20):
    out = []
    o = df[["AvgH", "AvgD", "AvgA"]].apply(pd.to_numeric, errors="coerce")
    n_missing = int(o.isna().any(axis=1).sum())
    if n_missing:
        by = df.assign(_m=o.isna().any(axis=1)).groupby("season")["_m"].sum()
        by = by[by > 0]
        out.append(Issue("WARN", "odds_missing", n_missing, f"rows without full odds, by season: {by.to_dict()}"))
    le1 = (o <= 1.0).any(axis=1)
    if le1.any():
        out.append(Issue("ERROR", "odds_le_1", int(le1.sum()), "odds <= 1.0 (impossible decimal odds)"))
    book = (1 / o).sum(axis=1, min_count=3)   # min_count: a row with NaN odds must stay NaN, not become 0.0
    odd = book.notna() & ((book < lo) | (book > hi))
    if odd.any():
        out.append(Issue("WARN", "overround_range", int(odd.sum()),
                         f"sum of 1/odds outside [{lo}, {hi}]: {_sample(df[odd].assign(book=book[odd].round(3)), ['Date','HomeTeam','AwayTeam','book'])}"))
    return out


def check_shots(df):
    out = []
    for hs, hst, side in (("HS", "HST", "home"), ("AS", "AST", "away")):
        a, b = pd.to_numeric(df[hs], errors="coerce"), pd.to_numeric(df[hst], errors="coerce")
        bad = b > a
        if bad.any():
            out.append(Issue("WARN", f"{hst}_gt_{hs}", int(bad.sum()),
                             f"impossible value in the source data ({side} shots on target > shots; does not break the pipeline): {_sample(df[bad], ['Date','HomeTeam','AwayTeam',hs,hst])}"))
        neg = (a < 0) | (b < 0)
        if neg.any():
            out.append(Issue("ERROR", f"{side}_shots_negative", int(neg.sum()), "negative shot counts"))
    partial = df["HST"].isna() != df["AST"].isna()
    if partial.any():
        out.append(Issue("WARN", "HST_AST_partial", int(partial.sum()), "only one of HST/AST is present"))
    miss = df[["HST", "AST"]].isna().any(axis=1)
    if miss.any():
        by = df.assign(_m=miss).groupby("season")["_m"].agg(["sum", "size"])
        by = by[by["sum"] > 0]
        out.append(Issue("INFO", "shots_missing", int(miss.sum()),
                         "rows with HST or AST missing, by season: " + ", ".join(f"{s}={int(r['sum'])}/{int(r['size'])}" for s, r in by.iterrows())))
    return out


def check_season_shape(df):
    """Teams per season and matches per season. A complete round-robin has N*(N-1) matches."""
    out = []
    rows = []
    for s, g in df.groupby("season"):
        teams = set(g["HomeTeam"]) | set(g["AwayTeam"])
        n = len(teams)
        full = n * (n - 1)
        rows.append((s, n, len(g), full))
        if len(g) > full:
            out.append(Issue("ERROR", "season_too_many_matches", len(g), f"season {s}: {len(g)} matches but {n} teams allow at most {full}"))
        elif len(g) < full:
            out.append(Issue("INFO", "season_incomplete", len(g), f"season {s}: {len(g)} of {full} matches ({n} teams) - partial/current season or missing rows"))
        else:
            hv = g.groupby("HomeTeam").size()
            av = g.groupby("AwayTeam").size()
            if not (hv == n - 1).all() or not (av == n - 1).all():
                out.append(Issue("ERROR", "season_unbalanced", len(g), f"season {s}: right match count but some team has wrong home/away count"))
    return out


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def check_team_names(df):
    out = []
    names = pd.Series(sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])), dtype=str)
    ws = names[names != names.str.strip()]
    if len(ws):
        out.append(Issue("ERROR", "team_whitespace", len(ws), f"leading/trailing spaces: {ws.tolist()[:5]}"))
    groups = {}
    for nme in names:
        groups.setdefault(_norm(nme), []).append(nme)
    clash = {k: v for k, v in groups.items() if len(v) > 1}
    if clash:
        out.append(Issue("ERROR", "team_name_variants", len(clash), f"names that differ only in case/punctuation: {list(clash.values())[:5]}"))
    # a team that exists in only 1-2 seasons with few matches may be a spelling variant of another
    seasons_per_team = pd.concat([df[["season", "HomeTeam"]].rename(columns={"HomeTeam": "T"}),
                                  df[["season", "AwayTeam"]].rename(columns={"AwayTeam": "T"})]).groupby("T")["season"].nunique()
    rare = seasons_per_team[seasons_per_team == 1]
    out.append(Issue("INFO", "teams_seen_one_season", len(rare), f"{len(rare)} of {len(names)} team names appear in exactly one season (normal for short stays; "
                                                               f"check by eye for spelling variants): {sorted(rare.index.tolist())[:12]}"))
    return out


def check_fixtures(fx, matches):
    """fixtures: DataFrame from fixtures.csv (Div, Date, Time, HomeTeam, AwayTeam). Checks names against the matches table."""
    out = []
    need = ["Div", "Date", "Time", "HomeTeam", "AwayTeam"]
    miss = [c for c in need if c not in fx.columns]
    if miss:
        return [Issue("ERROR", "fixtures_columns", len(miss), f"missing columns {miss}")]
    d = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    if d.isna().any():
        out.append(Issue("ERROR", "fixtures_date_parse", int(d.isna().sum()), "unparseable fixture dates"))
    t = fx["Time"].astype(str)
    badt = ~t.str.match(r"^\d{1,2}:\d{2}$")
    if badt.any():
        out.append(Issue("WARN", "fixtures_time", int(badt.sum()), f"missing/odd Time (treated as 00:00 by the logger): {fx.loc[badt, 'Time'].unique()[:5].tolist()}"))
    known = set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    cur = matches[matches["season"] == matches["season"].max()]
    current = set(cur["HomeTeam"]) | set(cur["AwayTeam"])
    for col in ("HomeTeam", "AwayTeam"):
        unknown = fx.loc[~fx[col].isin(known), col].unique().tolist()
        if unknown:
            out.append(Issue("ERROR", f"fixtures_unknown_{col}", len(unknown), f"never seen in the matches file (spelling mismatch?): {unknown}"))
        notcur = sorted(set(fx.loc[fx[col].isin(known) & ~fx[col].isin(current), col]))
        if notcur:
            out.append(Issue("WARN", f"fixtures_not_current_{col}", len(notcur), f"known historically but not in the latest season of the matches file (predict_match would refuse): {notcur}"))
    dup = fx.duplicated(["Date", "HomeTeam", "AwayTeam"], keep=False)
    if dup.any():
        out.append(Issue("ERROR", "fixtures_duplicate", int(dup.sum()), "duplicate fixture rows"))
    return out


def run_matches_checks(df, today=None):
    out = []
    out += check_columns(df)
    if out:                      # no point continuing without the columns
        return out
    for fn in (check_dates, check_season_labels, check_duplicates, check_results, check_odds,
               check_shots, check_season_shape, check_team_names):
        out += fn(df, today) if fn is check_dates else fn(df)
    return out