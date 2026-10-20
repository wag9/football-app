# audit_model.py -- READ-ONLY audit of the model/validation pipeline.
# Writes nothing to disk. Run:  python audit_model.py [E0|SP1|D1|I1|F1]
#
# What it checks (each section prints its own result):
#   1. Row sets: main.py base_df vs all rows, per season and per walk-forward fold
#   2. Fake zeros: how many rows get sot_diff = 0 only because shots data was NaN
#   3. Train/serve parity on real data (current shots.py vs the OLD serving rule)
#   4. Diagnostic walk-forward: Elo vs Elo+SoT on different row sets
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from data import load_matches
from elo import add_elo_features
from shots import WINDOW, MIN_PERIODS, SCALE, add_sot_feature, team_shots_form, _team_log
from model import K, HOME_ADV, NEW_START
from leagues import LEAGUES, DEFAULT

LABELS = ["A", "D", "H"]
FOLDS = [
    ("2023-08-01", "2024-08-01"),
    ("2024-08-01", "2025-08-01"),
    ("2025-08-01", "2026-08-01"),
]
PARITY_LAST_N = 150


# ---------------------------------------------------------------- helpers
def main_py_row_mask(df):
    """Boolean array (aligned to df rows by position): rows that survive main.py's dropna
    on the 12 old rolling features. Same logic as main.py steps 2-7."""
    d = df.copy().reset_index(drop=True)
    d["_pos"] = np.arange(len(d))

    home = d[["Date", "HomeTeam", "FTHG", "FTAG", "FTR"]].rename(
        columns={"HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA"})
    home["Pts"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})
    home["Venue"] = "Home"
    away = d[["Date", "AwayTeam", "FTAG", "FTHG", "FTR"]].rename(
        columns={"AwayTeam": "Team", "FTAG": "GF", "FTHG": "GA"})
    away["Pts"] = away["FTR"].map({"A": 3, "D": 1, "H": 0})
    away["Venue"] = "Away"
    team_df = pd.concat([home, away]).sort_values(["Team", "Date"]).reset_index(drop=True)

    for col in ["Pts", "GF", "GA"]:
        team_df[f"roll_{col.lower()}"] = team_df.groupby("Team")[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())

    ho = team_df[team_df["Venue"] == "Home"].copy()
    ho["home_ppg"] = ho.groupby("Team")["Pts"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
    ao = team_df[team_df["Venue"] == "Away"].copy()
    ao["away_ppg"] = ao.groupby("Team")["Pts"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())

    roll_cols = ["Date", "Team", "roll_pts", "roll_gf", "roll_ga"]
    d = d.merge(team_df[roll_cols], left_on=["Date", "HomeTeam"], right_on=["Date", "Team"],
                how="left").rename(columns={"roll_pts": "home_roll_pts", "roll_gf": "home_roll_gf",
                                            "roll_ga": "home_roll_ga"}).drop(columns=["Team"])
    d = d.merge(team_df[roll_cols], left_on=["Date", "AwayTeam"], right_on=["Date", "Team"],
                how="left").rename(columns={"roll_pts": "away_roll_pts", "roll_gf": "away_roll_gf",
                                            "roll_ga": "away_roll_ga"}).drop(columns=["Team"])
    d = d.merge(ho[["Date", "Team", "home_ppg"]], left_on=["Date", "HomeTeam"],
                right_on=["Date", "Team"], how="left").drop(columns=["Team"])
    d = d.merge(ao[["Date", "Team", "away_ppg"]], left_on=["Date", "AwayTeam"],
                right_on=["Date", "Team"], how="left").drop(columns=["Team"])

    if len(d) != len(df):
        print(f"  WARNING: merges changed row count {len(df)} -> {len(d)} (duplicate Date/Team keys?)")

    d["pts_diff"] = d["home_roll_pts"] - d["away_roll_pts"]
    d["gf_diff"] = d["home_roll_gf"] - d["away_roll_gf"]
    d["ga_diff"] = d["home_roll_ga"] - d["away_roll_ga"]
    d["ppg_diff"] = d["home_ppg"] - d["away_ppg"]
    cols = ["home_roll_pts", "away_roll_pts", "home_ppg", "away_ppg", "pts_diff", "gf_diff",
            "ga_diff", "ppg_diff", "home_roll_gf", "home_roll_ga", "away_roll_gf", "away_roll_ga"]
    keep = d.loc[d[cols].notna().all(axis=1), "_pos"].values
    mask = np.zeros(len(df), dtype=bool)
    mask[keep] = True
    return mask


def sot_real_flags(df):
    """True where BOTH teams have a real rolling SoT value (i.e. sot_diff is not a fake 0)."""
    st = _team_log(df)
    st["roll"] = st.groupby("Team")["net"].transform(
        lambda x: x.shift(1).rolling(window=WINDOW, min_periods=MIN_PERIODS).mean())
    hr = st[st["side"] == "h"].set_index("idx")["roll"].reindex(df.index)
    ar = st[st["side"] == "a"].set_index("idx")["roll"].reindex(df.index)
    return (hr.notna() & ar.notna()).values


def fold_masks(dates):
    out = []
    for start, end in FOLDS:
        tr = (dates < pd.Timestamp(start))
        te = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
        out.append((tr, te))
    return out


def walk_forward(fr, cols, row_mask):
    """Same as main.py walk_forward but restricted to row_mask. Returns (scores, n_train, n_test)."""
    scores, ntr, nte = [], [], []
    for tr, te in fold_masks(fr["Date"].values):
        tr = tr & row_mask
        te = te & row_mask
        if tr.sum() == 0 or te.sum() == 0:
            scores.append(float("nan")); ntr.append(int(tr.sum())); nte.append(int(te.sum()))
            continue
        m = LogisticRegression(max_iter=1000)
        m.fit(fr.loc[tr, cols], fr.loc[tr, "target"])
        scores.append(log_loss(fr.loc[te, "target"], m.predict_proba(fr.loc[te, cols]), labels=LABELS))
        ntr.append(int(tr.sum())); nte.append(int(te.sum()))
    return scores, ntr, nte


def old_serving_form(df):
    """OLD rule (before the fix): drop NaN first, then take the last WINDOW."""
    st = _team_log(df)
    form = {}
    for team, g in st.groupby("Team"):
        valid = g["net"].dropna().tail(WINDOW)
        form[team] = float(valid.mean()) if len(valid) >= MIN_PERIODS else np.nan
    return form


def serve_value(form, home, away):
    v = (form.get(home, np.nan) - form.get(away, np.nan)) * SCALE
    return 0.0 if np.isnan(v) else v


# ---------------------------------------------------------------- main
def main():
    league = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    cfg = LEAGUES[league]
    df = load_matches(cfg["matches"])
    n = len(df)
    print(f"=== audit_model.py | league {league} | rows loaded: {n} ===")
    print(f"constants: WINDOW={WINDOW} MIN_PERIODS={MIN_PERIODS} SCALE={SCALE} K={K} HA={HOME_ADV} NEW={NEW_START}")
    print("dates sorted:", df["Date"].is_monotonic_increasing,
          "| index unique:", df.index.is_unique, "\n")

    base_mask = main_py_row_mask(df)
    real = sot_real_flags(df)
    dates = df["Date"].values

    # ---- 1. row sets
    print("[1] ROW SETS")
    print(f"  all rows: {n} | main.py base_df rows: {int(base_mask.sum())} | dropped by dropna: {int((~base_mask).sum())}")
    if "season" in df.columns:
        t = pd.DataFrame({"season": df["season"].values, "all": 1, "base_df": base_mask.astype(int)})
        print(t.groupby("season").sum().to_string())
    for i, (tr, te) in enumerate(fold_masks(dates), 1):
        print(f"  fold {i}: test all={int(te.sum())} | test base_df={int((te & base_mask).sum())} "
              f"| train all={int(tr.sum())} | train base_df={int((tr & base_mask).sum())}")
    print("  NOTE: this covers the Elo/SoT walk-forward only. The Poisson '380 vs 374' claim lives in "
          "poisson.py, which this script does not touch -> still UNVERIFIED.\n")

    # ---- 2. fake zeros
    print("[2] FAKE-ZERO sot_diff (shots NaN or fewer than MIN_PERIODS valid matches)")
    hst_missing = df["HST"].isna() | df["AST"].isna()
    print(f"  rows with HST or AST missing: {int(hst_missing.sum())} of {n}")
    print(f"  rows where sot_diff is a fake 0: {int((~real).sum())} of {n} ({(~real).mean():.1%})")
    if "season" in df.columns:
        t = pd.DataFrame({"season": df["season"].values, "fake_zero": (~real).astype(int), "rows": 1})
        g = t.groupby("season").sum()
        g["share"] = (g["fake_zero"] / g["rows"]).round(3)
        print(g.to_string())
    for i, (tr, te) in enumerate(fold_masks(dates), 1):
        print(f"  fold {i}: fake zeros in train={int((tr & ~real).sum())}/{int(tr.sum())} "
              f"| in test={int((te & ~real).sum())}/{int(te.sum())}")
    print()

    # ---- 3. train/serve parity
    print(f"[3] TRAIN/SERVE PARITY on the last {PARITY_LAST_N} matches")
    sot_train = add_sot_feature(df).values
    diffs_new, diffs_old = [], []
    for p in range(max(0, n - PARITY_LAST_N), n):
        sub = df.iloc[:p]
        row = df.iloc[p]
        h, a = row["HomeTeam"], row["AwayTeam"]
        diffs_new.append(abs(serve_value(team_shots_form(sub), h, a) - sot_train[p]))
        diffs_old.append(abs(serve_value(old_serving_form(sub), h, a) - sot_train[p]))
    diffs_new, diffs_old = np.array(diffs_new), np.array(diffs_old)
    print(f"  CURRENT shots.py : max |serve - train| = {diffs_new.max():.6f} | rows differing (>1e-9): {int((diffs_new > 1e-9).sum())}")
    print(f"  OLD serving rule : max |serve - train| = {diffs_old.max():.6f} | rows differing (>1e-9): {int((diffs_old > 1e-9).sum())}")
    cur_new, cur_old = team_shots_form(df), old_serving_form(df)
    changed = [t for t in cur_new
               if not (np.isnan(cur_new[t]) and np.isnan(cur_old[t])) and
               (np.isnan(cur_new[t]) != np.isnan(cur_old[t]) or abs(cur_new[t] - cur_old[t]) > 1e-9)]
    print(f"  teams whose CURRENT serving form changed because of the fix: {len(changed)} of {len(cur_new)}")
    for t in sorted(changed)[:20]:
        print(f"    {t}: old={cur_old[t]:.3f} new={cur_new[t]:.3f}")
    print("  NOTE: Elo parity (ratings used at serve time vs elo_diff at train time) is NOT checked here.\n")

    # ---- 4. diagnostic walk-forward
    print("[4] DIAGNOSTIC WALK-FORWARD (log loss per fold, lower is better)")
    df_elo, _ = add_elo_features(df, k=K, home_adv=HOME_ADV, new_team_start=NEW_START, return_ratings=True)
    fr = pd.DataFrame({
        "Date": df["Date"].values,
        "target": df["FTR"].values,
        "elo_diff": df_elo["elo_diff"].values,
        "sot_diff": add_sot_feature(df).values,
    })
    sets = [
        ("A: base_df rows, fillna(0)  [= main.py / results.md]", base_mask),
        ("B: all rows, fillna(0)      [= production / validate_prod]", np.ones(n, dtype=bool)),
        ("C: real-shots rows only (all rows)", real),
        ("D: real-shots rows only AND base_df", real & base_mask),
    ]
    for name, mask in sets:
        s_e, ntr, nte = walk_forward(fr, ["elo_diff"], mask)
        s_s, _, _ = walk_forward(fr, ["elo_diff", "sot_diff"], mask)
        delta = [b - a for a, b in zip(s_e, s_s)]
        print(f"  {name}")
        print(f"    n_train per fold: {ntr} | n_test per fold: {nte}")
        print(f"    Elo only : {[round(x, 4) for x in s_e]} | avg {np.nanmean(s_e):.4f}")
        print(f"    Elo+SoT  : {[round(x, 4) for x in s_s]} | avg {np.nanmean(s_s):.4f}")
        print(f"    SoT delta: {[round(x, 4) for x in delta]} | avg {np.nanmean(delta):+.4f}  (negative = SoT helps)")
    print("\nDone. Nothing was written to disk.")


if __name__ == "__main__":
    main()