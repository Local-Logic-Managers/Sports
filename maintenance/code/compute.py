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
    # 2026-10-02: refreshed for Week 4 (vs. Tennessee Titans, 10/4) off Wed/Thu practice-participation
    # reports (Yahoo Sports, RoundTable.io, ClutchPoints, SharpFootballAnalysis roundup -- official
    # Friday designations hadn't posted yet as of this refresh). Nnamdi Madubuike and Teddye Buchanan
    # are off the injury report entirely (back to full health) -- removed. Zay Flowers and Marlon
    # Humphrey were upgraded to full practice Thursday and are expected to play -- removed from the
    # questionable list. New: Jovaughn Gwyn (OL) reportedly facing a multi-week absence (fractured
    # fibula per one report) and Ethan Pocic (C) DNP both Wed/Thu with a knee issue -- notable since
    # both are center depth behind the already-season-ending Danny Pinter. Lamar Jackson (back) was
    # limited Wed but full Thu with no game designation (HC Minter called it "minor") -- not modeled
    # as an injury adjustment since he's expected to play at full snap count.
    # 2026-10-05: refreshed for Week 5 (at Atlanta Falcons, 10/11) off this week's reporting
    # (NBC Sports, baltimoreravens.com, The Baltimore Banner, Roundtable.io). Ethan Pocic upgraded
    # out -> placed on IR alongside Gwyn (confirmed multi-source), not just a practice-report
    # questionable anymore -- Vega Ioane now stepping in at center. Lamar Jackson (back) and Zay
    # Flowers (hamstring) both back to full practice participation / reporting healthy -- no
    # longer modeled. Everything else carried forward unchanged (no fresh Week 5 report had
    # dropped yet as of this refresh; official Wed/Thu practice reports still to come).
    # 2026-10-07: new development since the 10-05 refresh -- Lamar Jackson suffered a LEFT ANKLE
    # SPRAIN late in the Week 4 win over Tennessee (10/4, landed on by a defender), left in a
    # protective boot, had an MRI Monday 10/6, and HC Jesse Minter was noncommittal on his Sunday
    # status ("I really don't think it's going to be crazy long-term, but we'll see" -- 10/5
    # quotes via Newsweek/Field Level Media). This supersedes last refresh's cleared "back" note --
    # that was about the PRIOR week's issue; the ankle happened afterward, in-game. No Wednesday
    # 10/7 practice-participation report had posted yet as of this refresh (checked multiple
    # sources) -- modeled conservatively as questionable pending Thursday/Friday's official word;
    # revisit once a real designation drops (backup Tyler Huntley would start if Jackson sits).
    "Baltimore Ravens": {
        "injuries": [
            {"pos": "C", "severity": "out"},              # Danny Pinter, out for the season
            {"pos": "WR", "severity": "out"},              # Ja'Kobi Lane, IR (wrist surgery), out for the season
            {"pos": "C", "severity": "out"},               # Jovaughn Gwyn, fractured fibula, IR, ~2-3 months
            {"pos": "C", "severity": "out"},               # Ethan Pocic, knee, placed on IR
            {"pos": "QB", "severity": "questionable"},     # Lamar Jackson, NEW left ankle sprain (Week 4 game), no designation yet
            {"pos": "OT", "severity": "questionable"},     # Ronnie Stanley, toe, trending toward full participation
            {"pos": "OLB", "severity": "questionable"},    # Trey Hendrickson, finger (multi-fracture per reports)
            {"pos": "G", "severity": "questionable"},      # John Simpson, groin, limited
            {"pos": "CB", "severity": "questionable"},     # T.J. Tampa, knee, no fresh update carried forward
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
# Real current market lines for the open Week 5 slate (DraftKings via statsniper.com, 2026-10-05) --
# keyed by (away, home). The full Week 4 slate above has been played/graded and is no longer open,
# so last run's Week 4 lines are retired. A game not in this dict just doesn't get a
# market_odds/value_bet/blended_prob block attached -- populated opportunistically, not every fixture.
NFL_MARKET_ODDS = {
    # Line moved toward Atlanta since the 10-05 snapshot (Ravens -192/+160) --
    # 10-06 median-of-books has Ravens as a smaller favorite, likely reflecting
    # the Gwyn/Pocic O-line IR moves. Range across books: Ravens -119 to -154,
    # Falcons +120 to +130, spread Ravens -2.5 to -3 (6 of 10 books at -3), total ~45-45.5.
    # 2026-10-07 re-check: scoresandstats.com still has Ravens favored (-154/+130, spread -2.5
    # to -3, total 45-45.5) -- essentially flat vs. 10-06. One other source (statsniper.com,
    # undated) showed a conflicting flip to Falcons -3.5 favorite, plausibly tied to Lamar
    # Jackson's new ankle-sprain uncertainty (see NFL_INJURY_TURNOVER) -- not used here pending a
    # confirmed/dated line; keeping Ravens as the modeled favorite but this is worth rechecking
    # once Thursday/Friday's official injury designation and fresh closing lines are out.
    ("Baltimore Ravens", "Atlanta Falcons"): {"home_ml": "+120", "away_ml": "-140", "book": "Median of books via statsniper.com/scoresandstats.com/bettoredge.com (2026-10-06); re-checked 2026-10-07, scoresandstats.com unchanged at Ravens -154/+130 -- see note above on a conflicting unconfirmed signal"},
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
    "status": "v2 ratings (win-total baseline + injury/turnover/regime-change adjustments) — Weeks 1-4 complete; Week 5 slate below (Oct 8-12)",
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
results["leagues"]["mlb"] = {
    "label": "MLB",
    "status": "2026 regular season complete (ended Sept 27) — ratings above are final-season power ratings. Postseason underway: Division Series as of Oct 7 (all four series have a potential series-clinching or elimination game tonight) — in the NLDS, the Dodgers lead the Braves 2-1 after a 3-1 Game 3 win on Oct 6 and can clinch an NLCS berth in Game 4 tonight, while the Brewers lead the Padres 2-1 after San Diego staved off elimination with a 4-3 Game 3 win on Oct 6, forcing a Game 4 tonight; in the ALDS, the White Sox lead the Guardians 2-0 and can complete a sweep to reach the ALCS in Game 3 tonight at Cleveland, and the Rays lead the Yankees 2-0 and can likewise sweep into the ALCS in Game 3 tonight at Yankee Stadium, an elimination game for New York — after a Wild Card round that saw the White Sox post their first postseason series win in 20+ years. Bracket-style postseason scheduling isn't modeled by this pipeline yet, so no new picks are generated until the format is extended.",
    "ratings": mlb_ratings, "games": mlb_games,
}

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
    # Updated 2026-10-07: DraftKings via gainblers.com (for the 2026-10-10 fixture) has Arsenal
    # ticking up slightly from the 10-04 Bet365 line (67.1/20.6/12.3 -> 68.4/19.1/12.5).
    ("Leeds United", "Arsenal"): {"home_win_prob": 0.684, "draw_prob": 0.191, "away_win_prob": 0.125, "book": "DraftKings via gainblers.com (2026-10-07), vig removed; prior Bet365/kickoff.co.uk (2026-10-04) had 67.1/20.6/12.3"},
    # Man City @ Liverpool, 2026-10-11: book treats it as a dead-even coin flip (both sides priced identically).
    # Re-checked 2026-10-07: betus.com.pa line unchanged (still dated 2026-10-05, same devigged numbers).
    ("Manchester City", "Liverpool"): {"home_win_prob": 0.371, "draw_prob": 0.258, "away_win_prob": 0.371, "book": "betus.com.pa (2026-10-06, re-checked 2026-10-07, unchanged), vig removed"},
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
