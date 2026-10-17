# leagues.py
DEFAULT = "E0"

LEAGUES = {
    "E0": {"name": "Premier League", "matches": "data/all_matches.csv",
           "log": "data/predictions_log.csv"},
    "SP1": {"name": "La Liga", "matches": "data/laliga_matches.csv",
            "log": "data/laliga_predictions_log.csv"},
}