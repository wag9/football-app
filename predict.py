# predict.py  -- compatibility wrapper (upcoming.py aur team_view.py ke liye)
from model import train, predict_match

_state = train()
df = _state["df"]
ratings = _state["ratings"]
current = _state["current"]


def predict(home, away, verbose=True):
    p = predict_match(_state, home, away)
    if verbose:
        print(f"{home} (Elo {ratings[home]:.0f}) vs {away} (Elo {ratings[away]:.0f})")
        print(f"  Home win: {p['H']:.0%} | Draw: {p['D']:.0%} | Away win: {p['A']:.0%}")
    return p


if __name__ == "__main__":
    top = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("Top 8 Elo:", [(t, round(r)) for t, r in top], "\n")
    predict("Arsenal", "Liverpool")