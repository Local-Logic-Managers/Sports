import csv, json, os, sys
from datetime import date, datetime
sys.path.insert(0, "/home/claude/sportspredict")
from engine import win_prob, make_pick, american_to_decimal, prob_to_american_odds, confidence_tier, wager_outcome

DATA = "/home/claude/sportspredict/data"
TODAY = date.today().isoformat()

# ---------- "should I bet on the Ravens this week?" verdict logic ----------
# Deliberately coarse and honest, reusing the same confidence tiers as the rest of
# the site (engine.CONFIDENCE_BANDS: Strong>=.80, Moderate>=.65, Lean>=.55, else
# Toss-up). A "large disagreement" vs. a real market line always overrides to a
# pass, regardless of which side the model favors -- per the same reasoning as
# the value-bet framing elsewhere: a big model/market gap is more likely a model
# blind spot (injuries, coaching change, roster turnover) than a real edge.

def bet_verdict(ravens_prob, value_bet=None):
    if value_bet and value_bet.get("large_disagreement"):
        return {
            "verdict": "PASS",
            "side": "none",
            "tone": "warn",
            "headline": "Model and market disagree too much to trust here",
            "reasoning": ("Our model and the actual sportsbook line are far enough apart ("
                          f"model {round(value_bet['model_prob']*100)}% vs. market-implied "
                          f"{round(value_bet['market_implied_prob']*100)}%) that the gap is more likely a blind "
                          "spot in our simple model than a real edge over Vegas. Sitting this one out is the "
                          "honest call."),
        }
    favored_ravens = ravens_prob >= 0.5
    tier_prob = ravens_prob if favored_ravens else (1 - ravens_prob)
    tier = confidence_tier(tier_prob)
    side = "ravens" if favored_ravens else "opponent"

    if tier == "Toss-up":
        return {
            "verdict": "PASS",
            "side": "none",
            "tone": "off",
            "headline": "Too close to call",
            "reasoning": f"The model has this game at {round(ravens_prob*100)}% for the Ravens — close enough to a coin flip that there's no real edge either way.",
        }
    if favored_ravens:
        label = "BET RAVENS" if tier in ("Strong", "Moderate") else "LEAN RAVENS"
        return {
            "verdict": label,
            "side": "ravens",
            "tone": "good" if tier == "Strong" else "ravens",
            "headline": f"Model likes the Ravens here ({tier.lower()} favorite)",
            "reasoning": f"The model gives the Ravens a {round(ravens_prob*100)}% win probability ({tier} tier) in this matchup" +
                         (f" — and {value_bet['read'].lower()}" if value_bet else ", though we don't have a real market line to compare against yet") + ".",
        }
    else:
        label = "FADE RAVENS" if tier in ("Strong", "Moderate") else "LEAN OPPONENT"
        return {
            "verdict": label,
            "side": "opponent",
            "tone": "away",
            "headline": f"Model favors the opponent here ({tier.lower()} favorite)",
            "reasoning": f"The model gives the Ravens only a {round(ravens_prob*100)}% win probability ({tier} tier against them) in this matchup" +
                         (f" — and {value_bet['read'].lower()}" if value_bet else ", though we don't have a real market line to compare against yet") + ".",
        }

def grade_ravens_calls(bet_log, actual_results):
    """actual_results: {week: actual_winner_team_name}. Mutates & returns bet_log,
    grading any open call whose week now has a known result. A call's recommended
    side "won" if that side actually won the game; PASS calls are graded for
    record-keeping (what would have happened) but excluded from the W-L record."""
    for call in bet_log:
        if call["graded"] or call["week"] not in actual_results:
            continue
        actual_winner = actual_results[call["week"]]
        ravens_won = (actual_winner == "Baltimore Ravens")
        if call["side"] == "ravens":
            side_won = ravens_won
        elif call["side"] == "opponent":
            side_won = not ravens_won
        else:
            side_won = None  # PASS -- no bet placed, nothing to grade as won/lost
        call["graded"] = True
        call["actual_winner"] = actual_winner
        call["side_won"] = side_won
    return bet_log

def build_track_record(bet_log):
    graded_actionable = [c for c in bet_log if c["graded"] and c["side"] != "none"]
    wins = sum(1 for c in graded_actionable if c["side_won"])
    losses = len(graded_actionable) - wins
    staked = 0.0
    returned = 0.0
    for c in graded_actionable:
        odds = prob_to_american_odds(c["side_prob"])
        out = wager_outcome(50, odds, c["side_won"])
        staked += out["stake"]
        returned += out["payout"]
    return {
        "total_calls": len(bet_log),
        "graded": sum(1 for c in bet_log if c["graded"]),
        "pending": sum(1 for c in bet_log if not c["graded"]),
        "pass_calls": sum(1 for c in bet_log if c["side"] == "none"),
        "actionable_graded": len(graded_actionable),
        "wins": wins,
        "losses": losses,
        "hit_rate": round(wins / len(graded_actionable), 3) if graded_actionable else None,
        "flat_50_staked": round(staked, 2),
        "flat_50_returned": round(returned, 2),
        "flat_50_profit": round(returned - staked, 2),
    }

def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))

with open(os.path.join(DATA, "model_params.json")) as f:
    PARAMS = json.load(f)["leagues"]["nfl"]

# Same static final-2025-derived ratings as compute.py (see that file's comment) --
# kept in sync here since compute_ravens.py needs the Ravens' + each opponent's rating.
#
# 2026-10-10 NOTE (tried and reverted a fix -- flagging for a real future fix instead): this
# static dict does NOT reflect compute.py's live, injury-aware v2 NFL ratings (nfl_ratings.json),
# so it currently can't see things like Lamar Jackson being ruled OUT for Week 5 -- the schedule
# below under-estimates how big an underdog that makes the Ravens this week. We tried wiring this
# script to read nfl_ratings.json directly, but that dict is a SINGLE current-day snapshot applied
# uniformly to the whole 18-week loop -- it would retroactively recompute Weeks 1-4 (already
# played and already locked into ravens_bet_log.json) using TODAY's injury picture, producing
# nonsense like "Week 1 vs Colts: 19.8%, large disagreement with the market" for a game the
# Ravens already won back in September. Reverted rather than ship that. A real fix needs
# per-week point-in-time ratings (snapshot the rating used at each week's logging time, don't
# recompute historical weeks from current state) -- a bigger change than belongs in an
# unattended daily refresh; flagged to Zak instead of guessing at it.
ratings = {"Denver Broncos": 3.706, "New England Patriots": 7.0, "Jacksonville Jaguars": 5.682, "Pittsburgh Steelers": 0.412, "Houston Texans": 4.488, "Buffalo Bills": 4.776, "Los Angeles Chargers": 1.153, "Indianapolis Colts": 2.224, "Baltimore Ravens": 1.071, "Miami Dolphins": -3.171, "Cincinnati Bengals": -3.212, "Kansas City Chiefs": 1.4, "Cleveland Browns": -4.118, "Las Vegas Raiders": -7.865, "New York Jets": -8.359, "Tennessee Titans": -7.988, "Seattle Seahawks": 7.865, "Chicago Bears": 1.071, "Philadelphia Eagles": 2.224, "Los Angeles Rams": 7.082, "San Francisco 49ers": 2.718, "Carolina Panthers": -2.841, "Tampa Bay Buccaneers": -1.276, "Atlanta Falcons": -1.976, "Green Bay Packers": 1.276, "Minnesota Vikings": 0.453, "Detroit Lions": 2.8, "Dallas Cowboys": -1.647, "New Orleans Saints": -3.171, "Washington Commanders": -3.912, "New York Giants": -2.388, "Arizona Cardinals": -5.476}

with open(os.path.join(DATA, "ravens_schedule.json")) as f:
    schedule = json.load(f)
with open(os.path.join(DATA, "ravens_data.json")) as f:
    ravens = json.load(f)

RAVENS = "Baltimore Ravens"
games = []
for g in schedule:
    if g.get("bye"):
        games.append(g)
        continue
    opp = g["opponent"]
    r_rav, r_opp = ratings.get(RAVENS, 0), ratings.get(opp, 0)
    home_adv = 0 if g.get("neutral") else PARAMS["home_adv"]
    if g["location"] == "home":
        p_ravens = win_prob(r_rav, r_opp, home_adv, PARAMS["sigma"])
    else:
        # ravens are away (or neutral): compute from the opponent's home perspective, then flip
        p_opp_home = win_prob(r_opp, r_rav, home_adv, PARAMS["sigma"])
        p_ravens = 1 - p_opp_home
    entry = {**g, "ravens_win_prob": round(p_ravens, 3), "model_odds": prob_to_american_odds(p_ravens)}
    if g.get("market_odds"):
        mkt = g["market_odds"]
        mkt_dec = american_to_decimal(mkt["ravens"])
        mkt_implied = round(1 / mkt_dec, 3)
        edge = round(p_ravens - mkt_implied, 3)
        if abs(edge) >= 0.15:
            read = ("Large disagreement with the market — more likely our simple model is missing context "
                    "(injuries, coaching change, roster turnover) than a real edge over Vegas's pricing.")
        elif edge > 0.02:
            read = "Model leans slightly more toward the Ravens than the market does."
        elif edge < -0.02:
            read = "Model leans slightly more toward the opponent than the market does."
        else:
            read = "Model roughly agrees with the market."
        entry["value_bet"] = {
            "market_implied_prob": mkt_implied,
            "model_prob": round(p_ravens, 3),
            "edge_pts": edge,
            "read": read,
            "large_disagreement": abs(edge) >= 0.15,
        }
    games.append(entry)

# season-long look: naive independent-games estimate of overall season win total (not a full simulation)
non_bye = [g for g in games if not g.get("bye")]
exp_wins = sum(g["ravens_win_prob"] for g in non_bye)

ravens["schedule"] = games
ravens["expected_wins_naive"] = round(exp_wins, 1)
ravens["games_count"] = len(non_bye)

# ---------- "should I bet this week" calls + track record ----------
# Idempotent: keep any already-logged (and possibly already-graded) call for a
# week untouched -- a verdict is locked in as of the day it was made, same as
# every other pick on the site being logged before the contest happens. Only
# add calls for weeks that aren't logged yet.
bet_log_path = os.path.join(DATA, "ravens_bet_log.json")
if os.path.exists(bet_log_path):
    with open(bet_log_path) as f:
        bet_log = json.load(f)
else:
    bet_log = []
logged_weeks = {c["week"] for c in bet_log}

for g in non_bye:
    if g["week"] in logged_weeks:
        continue
    v = bet_verdict(g["ravens_win_prob"], g.get("value_bet"))
    side_prob = g["ravens_win_prob"] if v["side"] == "ravens" else (1 - g["ravens_win_prob"] if v["side"] == "opponent" else g["ravens_win_prob"])
    bet_log.append({
        "week": g["week"], "opponent": g["opponent"], "date": g["date"], "location": g["location"],
        "ravens_win_prob": g["ravens_win_prob"], "model_odds": g["model_odds"],
        "verdict": v["verdict"], "side": v["side"], "side_prob": round(side_prob, 3),
        "tone": v["tone"], "headline": v["headline"], "reasoning": v["reasoning"],
        "logged_at": TODAY, "graded": False, "actual_winner": None, "side_won": None,
    })
bet_log.sort(key=lambda c: c["week"])

# Actual results for weeks played so far -- pass {week: actual_winner_team_name}.
# Week 1: Ravens beat the Colts 41-23 at Indianapolis (2026-09-13).
# Week 2: Ravens lost to the Saints 24-17 at home, blowing a 14-3 lead (2026-09-20).
# Week 3: Ravens beat the Cowboys 34-31 at a neutral site in Rio de Janeiro (2026-09-27).
# Week 4: Ravens beat the Titans 24-18 at home (2026-10-04).
# Week 5: Ravens beat the Falcons 30-23 in overtime at Atlanta (2026-10-11), Tyler Huntley starting
# for the injured Lamar Jackson.
bet_log = grade_ravens_calls(bet_log, {1: "Baltimore Ravens", 2: "New Orleans Saints", 3: "Baltimore Ravens", 4: "Baltimore Ravens", 5: "Baltimore Ravens"})

with open(bet_log_path, "w") as f:
    json.dump(bet_log, f, indent=2)

# "This week's" call = the soonest not-yet-graded, non-bye game from today.
upcoming = [c for c in bet_log if c["date"] >= TODAY]
this_week_call = min(upcoming, key=lambda c: c["date"]) if upcoming else None

ravens["bet_log"] = bet_log
ravens["bet_track_record"] = build_track_record(bet_log)
ravens["this_week_call"] = this_week_call

with open(os.path.join(DATA, "ravens_full.json"), "w") as f:
    json.dump(ravens, f, indent=2)

print(f"\nThis week's call: Week {this_week_call['week']} vs {this_week_call['opponent']} -> {this_week_call['verdict']}" if this_week_call else "\nNo upcoming call.")
print(f"Bet-call track record: {ravens['bet_track_record']}")

print(f"Expected wins (naive, independent-games estimate): {exp_wins:.1f} / {len(non_bye)}")
for g in games:
    if g.get("bye"):
        print(f"  Week {g['week']}: BYE")
        continue
    vb = f" | value: {g['value_bet']['read']} ({g['value_bet']['edge_pts']:+.1%})" if "value_bet" in g else ""
    print(f"  Week {g['week']} vs {g['opponent']} ({g['location']}): {g['ravens_win_prob']:.1%} [{g['model_odds']}]{vb}")
