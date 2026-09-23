import csv, math, json, os, sys, datetime
sys.path.insert(0, "/home/claude/sportspredict")
from engine import win_prob, shrink, make_pick, nfl_team_rating_v2, nfl_regime_change_sigma, attach_market

DATA = "/home/claude/sportspredict/data"
TODAY = datetime.date.today().isoformat()

def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))

with open(os.path.join(DATA, "model_params.json")) as f:
    PARAMS = json.load(f)

def P(league):
    return PARAMS["leagues"][league]

def blend_weight_for(league):
    """Fitted market-blend weight for a league (0-1), or None if it hasn't
    been backtested yet -- see model_params.json's market_blend_weight."""
    return PARAMS.get("market_blend_weight", {}).get(league)

def build_game(league, away, home, date, rh, ra, note=None, sigma_override=None, model_version=None):
    p = P(league)
    sigma = sigma_override if sigma_override is not None else p["sigma"]
    home_prob = win_prob(rh, ra, p["home_adv"], sigma)
    g = {"away": away, "home": home, "date": date, "home_win_prob": round(home_prob, 3)}
    if note:
        g["note"] = note
    if model_version is not None:
        g["model_version"] = model_version
    g.update(make_pick(away, home, home_prob))
    return g

# NOTE on design: league blocks below fall into two shapes.
#   (a) "live" leagues (MLB/EPL/MLS/PLL fixtures) read their ratings/schedule
#       from data files (CSVs) that a refresh step regenerates from real
#       current sources each run -- this file itself never hardcodes a game
#       date or matchup for these.
#   (b) "point-in-time" leagues (NFL/NCAAF Week 1, PLL's regular-season
#       ratings, NBA/NCAAB/UCL off-season placeholders) are genuinely static
#       until their season progresses -- real, already-scheduled/final data,
#       not stand-ins for a live fetch -- so they stay as literals here on
#       purpose. Revisit NFL/NCAAF once Week 1 has been played.

results = {"generated_at": TODAY, "model_version": PARAMS["version"], "leagues": {}}

# ---------------- NFL (v2: win-total-baseline ratings + situational adjustments) ----------------
# v2 replaces v1's point-diff-carryover ratings, which backtested at Brier
# 0.263 on real 2024 games (worse than a coin flip) because they were never
# anchored to anything with real predictive content -- see "Team rating
# methodology v2" in sports-predictor-app.md for the full diagnosis. Baseline
# = each team's real 2026 sportsbook season win total run through
# nfl_win_total_baseline() (inside nfl_team_rating_v2); layered with injury/
# turnover adjustments where we have specific sourced data, and a sigma bump
# for any team in its first season under a new head coach. Every NFL game
# built here is tagged model_version 2 -- already-logged v1 picks (e.g. the
# locked Week 1 slate) are a separate, untouched record; this only affects
# what gets computed/logged from here on.

NFL_SEASON_WIN_TOTALS = {  # 2026 season win totals, sourced via WebSearch/WebFetch (FOX Sports/DraftKings-derived), 2026-09-07
    "Arizona Cardinals": 3.5, "Atlanta Falcons": 6.5, "Baltimore Ravens": 11.5,
    "Buffalo Bills": 10.5, "Carolina Panthers": 7.5, "Chicago Bears": 9.5,
    "Cincinnati Bengals": 10.5, "Cleveland Browns": 5.5, "Dallas Cowboys": 9.5,
    "Denver Broncos": 9.5, "Detroit Lions": 10.5, "Green Bay Packers": 9.5,
    "Houston Texans": 9.5, "Indianapolis Colts": 7.5, "Jacksonville Jaguars": 8.5,
    "Kansas City Chiefs": 10.5, "Las Vegas Raiders": 5.5, "Los Angeles Chargers": 9.5,
    "Los Angeles Rams": 11.5, "Miami Dolphins": 4.5, "Minnesota Vikings": 8.5,
    "New England Patriots": 10.5, "New Orleans Saints": 7.5, "New York Giants": 7.5,
    "New York Jets": 5.5, "Philadelphia Eagles": 10.5, "Pittsburgh Steelers": 8.5,
    "San Francisco 49ers": 9.5, "Seattle Seahawks": 10.5, "Tampa Bay Buccaneers": 8.5,
    "Tennessee Titans": 6.5, "Washington Commanders": 7.5,
}

NFL_NEW_HC_TEAMS = {  # first season under a new head coach, 2026 -- drives the regime-change sigma bump
    "Buffalo Bills", "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens",
    "Cleveland Browns", "Las Vegas Raiders", "Miami Dolphins", "New York Giants",
    "Pittsburgh Steelers", "Tennessee Titans",
}

# Injury/turnover adjustments where we have specific, sourced data (currently
# just the Ravens -- see ravens-data.json). Teams not listed here get 0 for
# both, an honest default: the win-total baseline already does nearly all the
# work per the Ravens diagnostic. Extend opportunistically as real injury/
# transaction news gets sourced for other teams.
NFL_INJURY_TURNOVER = {
    # 2026-09-21: refreshed for Week 3 off HC Jesse Minter's post-Week-2 press conference
    # (Flowers/Stanley/Madubuike specifically named as trending toward availability, final call
    # Wednesday 9/23 -- so downgraded from "out" to "questionable" rather than assumed clear) plus
    # the FantasyPros Week 3 injury roundup. Ja'Kobi Lane placed on IR (wrist surgery) -- out for
    # the season. Buchanan/Tampa had no fresh Week 3 update, so softened from Week 2's definitive
    # "out" to "questionable" rather than carried forward as still-out with no supporting news.
    "Baltimore Ravens": {
        "injuries": [
            {"pos": "C", "severity": "out"},              # Danny Pinter, out for the season
            {"pos": "WR", "severity": "out"},              # Ja'Kobi Lane, IR (wrist surgery), out for the season
            {"pos": "WR", "severity": "questionable"},     # Zay Flowers, hamstring, trending toward playing (final call 9/23)
            {"pos": "OT", "severity": "questionable"},     # Ronnie Stanley, toe, trending toward playing (final call 9/23)
            {"pos": "DT", "severity": "questionable"},     # Nnamdi Madubuike, neck, trending toward playing (final call 9/23)
            {"pos": "ILB", "severity": "questionable"},    # Teddye Buchanan, knee, no fresh Week 3 update
            {"pos": "CB", "severity": "questionable"},     # T.J. Tampa, knee, no fresh Week 3 update
            {"pos": "OLB", "severity": "questionable"},    # Trey Hendrickson, finger, recurring
            {"pos": "G", "severity": "questionable"},      # John Simpson, groin, recurring
        ],
        "departures": ["TE", "FB", "OT", "P", "S", "CB", "OLB", "WR"],  # Likely, Ricard, Faalele, Stout, Washington, Alexander, Oweh, C.Johnson
        "acquisitions": ["OLB", "S", "CB", "LB", "ILB"],                 # D.Jones, Gilman, White, Hendrickson, Barrett
    },
    "Carolina Panthers": {
        "injuries": [
            {"pos": "OT", "severity": "out"},  # Ikem Ekwonu, ruptured patellar tendon, out most/all season
            {"pos": "OT", "severity": "out"},  # Taylor Moton, blood clot in lung, out extended time
        ],
    },
    "Detroit Lions": {
        "injuries": [
            {"pos": "S", "severity": "out"},          # Brian Branch, ruptured Achilles, targeting Dec return
            {"pos": "S", "severity": "questionable"},  # Kerby Joseph, nagging knee, recovery timeline unclear
        ],
    },
    "Green Bay Packers": {
        "injuries": [
            {"pos": "EDGE", "severity": "out"},  # Micah Parsons, torn ACL, targeting a playoff return
            {"pos": "WR", "severity": "out"},    # Jayden Reed, neck/spinal injury (stretchered off Week 2), 2026-09-21
        ],
    },
    "Seattle Seahawks": {
        "injuries": [
            {"pos": "RB", "severity": "out"},          # Zach Charbonnet, torn ACL, out extended time
            {"pos": "RB", "severity": "questionable"},  # Jadarian Price, chest injury, 2026-09-21
        ],
    },
    "San Francisco 49ers": {
        "injuries": [
            {"pos": "TE", "severity": "questionable"},  # George Kittle, Achilles, recovery timeline unclear
            {"pos": "WR", "severity": "out"},           # De'Zhaun Stribling, ankle (IR, ~10wk), 2026-09-21
        ],
    },
    "Indianapolis Colts": {
        "injuries": [
            {"pos": "WR", "severity": "out"},  # Alec Pierce, left heel (prior surgical site), 2026-09-21
        ],
    },
    # 2026-09-21 additions (FantasyPros Week 3 injury roundup + team beat reporting), teams newly
    # entering the currently-tracked Week 3 slate.
    "Chicago Bears": {
        "injuries": [
            {"pos": "QB", "severity": "questionable"},  # Caleb Williams, hamstring strain, week-to-week per HC Ben Johnson
        ],
    },
    "Washington Commanders": {
        "injuries": [
            {"pos": "QB", "severity": "out"},  # Jayden Daniels, left elbow dislocation
        ],
    },
    "Philadelphia Eagles": {
        "injuries": [
            {"pos": "TE", "severity": "out"},           # Dallas Goedert, right knee
            {"pos": "RB", "severity": "questionable"},  # Saquon Barkley, left shoulder stinger, day-to-day
        ],
    },
    "Denver Broncos": {
        "injuries": [
            {"pos": "RB", "severity": "questionable"},  # J.K. Dobbins, hamstring
            {"pos": "RB", "severity": "out"},           # RJ Harvey, hamstring strain
        ],
    },
    "Minnesota Vikings": {
        "injuries": [
            {"pos": "RB", "severity": "out"},  # Jordan Mason, fractured right thumb (surgical, IR)
        ],
    },
    "Atlanta Falcons": {
        "injuries": [
            {"pos": "QB", "severity": "positive_return"},  # Michael Penix Jr., back from 2025 ACL tear, season debut expected
        ],
    },
    "Houston Texans": {
        "injuries": [
            {"pos": "WR", "severity": "out"},  # Nico Collins, grade 1 hamstring strain, likely out
        ],
    },
    "Las Vegas Raiders": {
        "injuries": [
            {"pos": "TE", "severity": "out"},  # Brock Bowers, meniscus trim (right knee), expected back soon
        ],
    },
    "Dallas Cowboys": {
        "injuries": [
            {"pos": "S", "severity": "out"},            # P.J. Locke, foot (non-contact)
            {"pos": "CB", "severity": "out"},           # Cobie Durant, hamstring
            {"pos": "ILB", "severity": "questionable"},  # Dee Winters, shoulder
            {"pos": "ILB", "severity": "questionable"},  # Jaishawn Barham, stinger
        ],
    },
}

nfl_ratings = {}
for team, win_total in NFL_SEASON_WIN_TOTALS.items():
    extra = NFL_INJURY_TURNOVER.get(team, {})
    nfl_ratings[team] = round(nfl_team_rating_v2(
        win_total,
        injuries=extra.get("injuries"),
        departures=extra.get("departures"),
        acquisitions=extra.get("acquisitions"),
    ), 3)

# 2026-09-08: was a hardcoded Week-1-only literal list; now reads from nfl_slate.csv
# (away,home,date -- same convention as mlb_slate.csv/epl_fixtures.csv/etc.) so adding Week 2+
# is a data update, not a code change. 2026-09-15: Week 1 is complete (all 16 games graded --
# see predictions-history.json) and nfl_slate.csv now holds the real Week 2 slate (Sept 17-21).
# log_predictions.py's append-only/no-duplicate logic still governs whether new rows here
# actually produce new logged picks.
nfl_week1 = [(r["away"], r["home"], r["date"]) for r in read_csv("nfl_slate.csv")]
# Real current market lines for the open Week 3 slate (FanDuel Research moneylines, sourced
# via WebSearch/WebFetch, 2026-09-21; Ravens game uses the more specific FanDuel line already
# tracked in ravens-schedule.json for consistency) -- keyed by (away, home). A game not in this
# dict just doesn't get a market_odds/value_bet/blended_prob block attached.
NFL_MARKET_ODDS = {
    ("Atlanta Falcons", "Green Bay Packers"): {"home_ml": "-270", "away_ml": "+220", "book": "FanDuel (2026-09-21)"},
    ("Los Angeles Chargers", "Buffalo Bills"): {"home_ml": "-360", "away_ml": "+290", "book": "FanDuel (2026-09-21)"},
    ("Carolina Panthers", "Cleveland Browns"): {"home_ml": "+124", "away_ml": "-146", "book": "FanDuel (2026-09-21)"},
    ("New York Jets", "Detroit Lions"): {"home_ml": "-335", "away_ml": "+270", "book": "FanDuel (2026-09-21)"},
    ("Houston Texans", "Indianapolis Colts"): {"home_ml": "+128", "away_ml": "-152", "book": "FanDuel (2026-09-21)"},
    ("New England Patriots", "Jacksonville Jaguars"): {"home_ml": "-156", "away_ml": "+132", "book": "FanDuel (2026-09-21)"},
    ("Kansas City Chiefs", "Miami Dolphins"): {"home_ml": "+540", "away_ml": "-770", "book": "FanDuel (2026-09-21)"},
    ("Tennessee Titans", "New York Giants"): {"home_ml": "-194", "away_ml": "+162", "book": "FanDuel (2026-09-21)"},
    ("Cincinnati Bengals", "Pittsburgh Steelers"): {"home_ml": "+144", "away_ml": "-172", "book": "FanDuel (2026-09-21)"},
    ("Seattle Seahawks", "Washington Commanders"): {"home_ml": "+245", "away_ml": "-300", "book": "FanDuel (2026-09-21)"},
    ("Arizona Cardinals", "San Francisco 49ers"): {"home_ml": "-460", "away_ml": "+360", "book": "FanDuel (2026-09-21)"},
    ("Minnesota Vikings", "Tampa Bay Buccaneers"): {"home_ml": "+102", "away_ml": "-120", "book": "FanDuel (2026-09-21)"},
    ("Las Vegas Raiders", "New Orleans Saints"): {"home_ml": "-168", "away_ml": "+142", "book": "FanDuel (2026-09-21)"},
    ("Baltimore Ravens", "Dallas Cowboys"): {"home_ml": "+134", "away_ml": "-158", "book": "FanDuel (2026-09-22)"},
    ("Los Angeles Rams", "Denver Broncos"): {"home_ml": "+118", "away_ml": "-138", "book": "FanDuel (2026-09-21)"},
    ("Philadelphia Eagles", "Chicago Bears"): {"home_ml": "+146", "away_ml": "-174", "book": "FanDuel (2026-09-21)"},
}

nfl_games = []
for a, h, d in nfl_week1:
    g = build_game(
        "nfl", a, h, d, nfl_ratings.get(h, 0), nfl_ratings.get(a, 0),
        sigma_override=nfl_regime_change_sigma(
            P("nfl")["sigma"],
            home_new_coach=h in NFL_NEW_HC_TEAMS,
            away_new_coach=a in NFL_NEW_HC_TEAMS,
        ),
        model_version=2,
    )
    market = NFL_MARKET_ODDS.get((a, h))
    if market:
        g = attach_market(g, market, blend_weight=blend_weight_for("nfl"))
    nfl_games.append(g)
results["leagues"]["nfl"] = {
    "label": "NFL",
    "status": "v2 ratings (win-total baseline + injury/turnover/regime-change adjustments) — Weeks 1-2 complete (all graded), Week 3 slate below (Sept 24-28, incl. Ravens-Cowboys in Rio de Janeiro)",
    "model_version": 2,
    "ratings": nfl_ratings, "games": nfl_games,
}

# ---------------- MLB (live: fresh standings + slate each refresh) ----------------
mlb_rows = read_csv("mlb_2026_standings.csv")
mlb_ratings = {}
for r in mlb_rows:
    w, l = int(r["w"]), int(r["l"])
    gp = w + l
    raw = int(r["run_diff"]) / gp
    mlb_ratings[r["team"]] = round(shrink(raw, gp, P("mlb")["shrink_k"]), 3)

mlb_slate = [(r["away"], r["home"], r["date"]) for r in read_csv("mlb_slate.csv")]
mlb_games = [build_game("mlb", a, h, d, mlb_ratings.get(h, 0), mlb_ratings.get(a, 0)) for a, h, d in mlb_slate]
results["leagues"]["mlb"] = {"label": "MLB", "status": "in season", "ratings": mlb_ratings, "games": mlb_games}

# ---------------- Premier League (live) ----------------
epl_rows = read_csv("epl_2026_27_standings.csv")
epl_ratings = {}
for r in epl_rows:
    gp = int(r["gp"])
    raw = int(r["gd"]) / gp if gp else 0
    epl_ratings[r["team"]] = round(shrink(raw, gp, P("epl")["shrink_k"]), 3)

# Real current market lines for specific EPL games where we have a clean 3-way
# consensus price (Covers.com) -- keyed by (away, home). Most EPL games won't be
# in this dict; it's populated opportunistically, not every fixture.
EPL_MARKET_ODDS = {
    ("Arsenal", "Aston Villa"): {"home_win_prob": 0.13, "draw_prob": 0.23, "away_win_prob": 0.65, "book": "Covers.com consensus"},
}

epl_fixtures = [(r["away"], r["home"], r["date"]) for r in read_csv("epl_fixtures.csv")]
epl_games = []
for a, h, d in epl_fixtures:
    g = build_game("epl", a, h, d, epl_ratings.get(h, 0), epl_ratings.get(a, 0))
    market = EPL_MARKET_ODDS.get((a, h))
    if market:
        g = attach_market(g, market, blend_weight=blend_weight_for("epl"))
    epl_games.append(g)
results["leagues"]["epl"] = {
    "label": "Premier League",
    "status": "in season — early-season ratings are heavily shrunk toward league average",
    "ratings": epl_ratings, "games": epl_games,
}

# ---------------- MLS (live) ----------------
mls_rows = read_csv("mls_2026_standings.csv")
mls_ratings = {}
for r in mls_rows:
    gp = int(r["gp"])
    raw = int(r["gd"]) / gp if gp else 0
    mls_ratings[r["team"]] = round(shrink(raw, gp, P("mls")["shrink_k"]), 3)

mls_fixtures = [(r["away"], r["home"], r["date"]) for r in read_csv("mls_fixtures.csv")]
mls_games = [build_game("mls", a, h, d, mls_ratings.get(h, 0), mls_ratings.get(a, 0)) for a, h, d in mls_fixtures]
results["leagues"]["mls"] = {"label": "MLS", "status": "in season", "ratings": mls_ratings, "games": mls_games}

# ---------------- PLL Lacrosse (regular-season ratings are final/static; fixtures are live) ----------------
pll_ratings = {
    "Philadelphia Waterdogs": 2.5, "Boston Cannons": 0.833, "Maryland Whipsnakes": -0.833,
    "New York Atlas": -2.5, "Utah Archers": 0.833, "California Redwoods": 0.0,
    "Denver Outlaws": 0.0, "Carolina Chaos": -0.833,
}
pll_fixtures = [(r["away"], r["home"], r["date"], r["note"]) for r in read_csv("pll_fixtures.csv")]
pll_games = [build_game("pll", a, h, d, pll_ratings.get(h, 0), pll_ratings.get(a, 0), note) for a, h, d, note in pll_fixtures]
results["leagues"]["pll"] = {
    "label": "PLL Lacrosse",
    "status": "2026 season concluded — Philadelphia Waterdogs defeated Denver Outlaws 14-4 in the Sept 20 championship (Sports Illustrated Stadium, Harrison NJ) for their 2nd PLL title. No further games until the 2027 season.",
    "ratings": pll_ratings, "games": pll_games,
}

# ---------------- NCAA Football (point-in-time: real Week 1 schedule, AP Top 25 preseason poll) ----------------
ap25 = ["Ohio State","Oregon","Georgia","Notre Dame","Texas","Indiana","Miami","Texas A&M",
        "Ole Miss","Oklahoma","LSU","Texas Tech","Alabama","BYU","USC","Michigan","Washington",
        "Penn State","SMU","Tennessee","Utah","Iowa","Houston","Louisville","Missouri"]
ncaaf_ratings = {team: round(max(0, 26 - (i + 1)) * 2.4, 3) for i, team in enumerate(ap25)}

# Week 1 (2026-09-05) has been played and graded (all 4 favorites covered — see
# predictions-history.json). The Week-2+ file-driven pipeline (CSV-based
# standings/fixtures like MLB/EPL/MLS) hasn't been built yet -- flagged for
# Zak rather than guessing at a new hardcoded slate with no real ranking
# update methodology behind it. Leaving games empty (rather than re-showing
# the finished Week 1 slate as if it were still upcoming) until that's built.
ncaaf_games = []
results["leagues"]["ncaaf"] = {
    "label": "NCAA Football",
    "status": "Week 1 complete (all graded) — Week 2+ needs a file-driven pipeline (not yet built); scoped to AP Top 25 preseason poll",
    "ratings": ncaaf_ratings, "games": ncaaf_games,
}

# ---------------- NBA (offseason, ratings only, static until 2026-27 tips off) ----------------
nba_ratings = {"Oklahoma City Thunder": 5.61, "San Antonio Spurs": 5.122, "Detroit Pistons": 4.634, "Denver Nuggets": 3.171, "Los Angeles Lakers": 2.927, "Boston Celtics": 3.659, "New York Knicks": 2.927, "Houston Rockets": 2.683, "Cleveland Cavaliers": 2.683, "Minnesota Timberwolves": 1.951, "Toronto Raptors": 1.22, "Atlanta Hawks": 1.22, "Portland Trail Blazers": 0.244, "Los Angeles Clippers": 0.244, "Philadelphia 76ers": 0.976, "Orlando Magic": 0.976, "Phoenix Suns": 0.976, "Charlotte Hornets": 0.732, "Miami Heat": 0.488, "Golden State Warriors": -0.976, "Milwaukee Bucks": -2.195, "Chicago Bulls": -2.439, "New Orleans Pelicans": -3.659, "Dallas Mavericks": -3.659, "Memphis Grizzlies": -3.902, "Sacramento Kings": -4.634, "Utah Jazz": -4.634, "Indiana Pacers": -5.366, "Washington Wizards": -5.854, "Brooklyn Nets": -5.122}

results["leagues"]["nba"] = {
    "label": "NBA",
    "status": "offseason — 2026-27 season tips off in October; ratings below are final 2025-26 power ratings (unregressed)",
    "ratings": nba_ratings, "games": [],
}

# ---------------- NCAA Basketball / Champions League (placeholders) ----------------
results["leagues"]["ncaab"] = {
    "label": "NCAA Basketball", "status": "offseason — season tips off in November 2026",
    "ratings": {}, "games": [],
}
results["leagues"]["ucl"] = {
    "label": "Champions League", "status": "league phase has not started — kicks off mid-September 2026",
    "ratings": {}, "games": [],
}

with open(os.path.join(DATA, "predictions.json"), "w") as f:
    json.dump(results, f, indent=2)

for lk, lv in results["leagues"].items():
    print(lk, lv["label"], "games:", len(lv["games"]), "teams rated:", len(lv["ratings"]))
    for g in lv["games"]:
        print("  ", g["away"], "@", g["home"], "->", g["home_win_prob"], "pick:", g["pick"], g["confidence"])
