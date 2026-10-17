# leagues.py
DEFAULT = "E0"

# key = football-data Div code; prefix = raw season files data/<prefix>2627.csv
LEAGUES = {
    "E0":  {"name": "Premier League", "prefix": "prem",
            "matches": "data/all_matches.csv",
            "log": "data/predictions_log.csv"},
    "SP1": {"name": "La Liga", "prefix": "laliga",
            "matches": "data/laliga_matches.csv",
            "log": "data/laliga_predictions_log.csv"},
    "D1":  {"name": "Bundesliga", "prefix": "bundesliga",
            "matches": "data/bundesliga_matches.csv",
            "log": "data/bundesliga_predictions_log.csv"},
    "I1":  {"name": "Serie A", "prefix": "seriea",
            "matches": "data/seriea_matches.csv",
            "log": "data/seriea_predictions_log.csv"},
    "F1":  {"name": "Ligue 1", "prefix": "ligue1",
            "matches": "data/ligue1_matches.csv",
            "log": "data/ligue1_predictions_log.csv"},
}