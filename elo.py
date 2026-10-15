# elo.py

def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update(ra, rb, actual_a, k=20):
    ea = expected_score(ra, rb)
    new_ra = ra + k * (actual_a - ea)
    new_rb = rb + k * ((1 - actual_a) - (1 - ea))
    return new_ra, new_rb


def goal_multiplier(goal_diff):
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def add_elo_features(df, k=30, home_adv=75, start=1500, reversion=1.0,
                     new_team_start=None, return_ratings=False, use_mov=False):
    if new_team_start is None:
        new_team_start = start

    ratings = {}
    score_map = {"H": 1.0, "D": 0.5, "A": 0.0}
    elo_home, elo_away = [], []
    first_season = df["season"].iloc[0]
    current_season = None

    def get_rating(team, season):
        if team in ratings:
            return ratings[team]
        return start if season == first_season else new_team_start

    for row in df.itertuples():
        if row.season != current_season:
            if current_season is not None and reversion != 1.0:
                for team in ratings:
                    ratings[team] = start + reversion * (ratings[team] - start)
            current_season = row.season

        rh = get_rating(row.HomeTeam, row.season)
        ra = get_rating(row.AwayTeam, row.season)

        elo_home.append(rh)
        elo_away.append(ra)

        actual = score_map[row.FTR]
        k_eff = k * goal_multiplier(row.FTHG - row.FTAG) if use_mov else k
        new_h, _ = update(rh + home_adv, ra, actual, k_eff)
        delta = new_h - (rh + home_adv)
        ratings[row.HomeTeam] = rh + delta
        ratings[row.AwayTeam] = ra - delta

    df = df.copy()
    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    if return_ratings:
        return df, ratings
    return df