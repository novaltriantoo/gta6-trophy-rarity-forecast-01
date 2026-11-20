"""Forecast how many GTA VI players earn each trophy by the target time.

Entry for make.fun's gta6-trophy-rarity-forecast-01. The program reads one
case file per run: every read the house made of the game's Exophase trophy
page before the case's checkpoint, each read holding the players tracked
and every trophy's earn rate (the percent of tracked players who earned
it). For each trophy it writes five quantiles of the earn rate at the
case's target, scored by pinball loss on the logit scale.

How the forecast is built, per trophy:

  - The reads are compressed to the distinct snapshots (Exophase refreshes
    its counts on its own schedule, so long runs of reads repeat one page).
  - Two trend views are fitted by weighted least squares against
    x = ln(1 + hours since release midnight), with weights that decay
    towards the recent reads, and each slope is shrunk towards zero by the
    fit's effective sample size, so a noisy short series barely moves:
      A. the rate itself, on the logit scale;
      B. the earner count, rate x players / 100, on the log scale, against
         a players-tracked forecast fitted the same way. This view sees
         the denominator: a trophy every current player holds is diluted
         as new players arrive, and a rare trophy's count can keep
         climbing while its rate looks flat.
  - The median is the average of the two views on the logit scale. The
    spread combines a random-walk term that grows with the horizon, the
    trend fit's residual scale, and the slope's own uncertainty carried
    over the extrapolation distance, widened again when the snapshots are
    few. Quantiles are the median plus z x spread on the logit scale.

Everything is a pure function of the case file, so two runs of the same
case write the same bytes. Standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# A printed 0.00 means the true rate is under 0.005; use the middle of
# that interval when a count is needed, and the scorer's own clamp when
# a logit is needed.
PRINT_FLOOR_MID = 0.0025
RATE_MIN = 0.01
RATE_MAX = 99.99

WALK_SIGMA = 0.55        # random-walk spread, logit units per sqrt day
DECAY_HOURS = 36.0       # recency weight decay constant, in hours
SHRINK_PRIOR = 5.0       # effective samples before a slope is half trusted
MAX_LOGIT_MOVE = 2.8     # cap on a trend's total logit move to the target
SPREAD_FLOOR = 0.30
SPREAD_CAP = 2.4


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def x_of_hours(hours):
    return math.log(1.0 + max(hours, -0.9))


def clamp_rate(rate):
    return min(RATE_MAX, max(RATE_MIN, rate))


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x.

    Returns (a, b, residual_sd, slope_se, n_eff). With too little signal
    the slope comes back 0 and the intercept the weighted mean.
    """
    sw = sum(ws)
    if sw <= 0:
        return (ys[0] if ys else 0.0), 0.0, 0.0, 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    sw2 = sum(w * w for w in ws)
    n_eff = sw * sw / sw2 if sw2 > 0 else 0.0
    if sxx <= 1e-12 or len(xs) < 2:
        return my, 0.0, 0.0, 0.0, n_eff
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n_eff - 2.0, 1.0)
    var = sum(w * r * r for r, w in zip(resid, ws)) / dof
    sd = math.sqrt(max(var, 0.0))
    slope_se = math.sqrt(max(var, 0.0) / sxx) if sxx > 0 else 0.0
    return a, b, sd, slope_se, n_eff


def shrunk_slope(slope, n_eff):
    return slope * n_eff / (n_eff + SHRINK_PRIOR)


def snapshots(history):
    """Compress the reads to the moments the page actually changed."""
    out = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append({
            "hours": hours_since_epoch(parse_time(read["at"])),
            "players": read.get("players_tracked") or 0,
            "rates": rates,
        })
    if history and (not out or out[-1]["hours"] != hours_since_epoch(parse_time(history[-1]["at"]))):
        last = history[-1]
        out.append({
            "hours": hours_since_epoch(parse_time(last["at"])),
            "players": last.get("players_tracked") or 0,
            "rates": last.get("rates") or [],
        })
    return out


def weights_for(snaps):
    last = snaps[-1]["hours"]
    return [math.exp(-(last - s["hours"]) / DECAY_HOURS) for s in snaps]


def forecast_players(snaps, ws, x_target, days):
    """Forecast the players-tracked count at the target."""
    last_players = max(snaps[-1]["players"], 1)
    xs = [x_of_hours(s["hours"]) for s in snaps]
    ys = [math.log(max(s["players"], 1)) for s in snaps]
    a, b, _sd, _se, n_eff = wls(xs, ys, ws)
    b = max(shrunk_slope(b, n_eff), 0.0)
    grown = math.exp(a + b * x_target)
    lo = float(last_players)
    hi = float(last_players) * 40.0
    return min(max(grown, lo), hi)


def forecast_trophy(series, x_target, dx, days, players_hat, few_snaps):
    """series: [(hours, rate, players)] over distinct snapshots, rate set.

    Returns the five quantiles for one trophy.
    """
    last_rate = series[-1][1]
    last_hours = series[-1][0]
    xs = [x_of_hours(s[0]) for s in series]
    ws = [math.exp(-(last_hours - s[0]) / DECAY_HOURS) for s in series]

    # View A: the rate on the logit scale.
    ys_a = [logit_pct(s[1]) for s in series]
    a_a, b_a, sd_a, se_a, n_a = wls(xs, ys_a, ws)
    move_a = shrunk_slope(b_a, n_a) * dx
    move_a = min(max(move_a, -MAX_LOGIT_MOVE), MAX_LOGIT_MOVE)
    logit_a = logit_pct(last_rate) + move_a if len(series) >= 2 else logit_pct(last_rate)

    # View B: the earner count against the players forecast.
    logit_b = None
    if len(series) >= 2 and players_hat > 0:
        ys_b = []
        for _h, rate, players in series:
            shown = rate if rate > 0 else PRINT_FLOOR_MID
            earners = shown * max(players, 1) / 100.0
            ys_b.append(math.log(earners + 0.5))
        a_b, b_b, _sd_b, _se_b, n_b = wls(xs, ys_b, ws)
        b_b = max(shrunk_slope(b_b, n_b), 0.0)
        earners_hat = math.exp(a_b + b_b * x_target) - 0.5
        last_earners = (last_rate if last_rate > 0 else PRINT_FLOOR_MID) \
            * max(series[-1][2], 1) / 100.0
        earners_hat = max(earners_hat, last_earners * 0.9)
        earners_hat = min(earners_hat, players_hat)
        logit_b = logit_pct(100.0 * earners_hat / players_hat)

    if logit_b is None:
        median_logit = logit_a
    else:
        median_logit = 0.5 * (logit_a + logit_b)

    spread = math.sqrt(
        (WALK_SIGMA * math.sqrt(days)) ** 2
        + max(sd_a, 0.15) ** 2
        + (se_a * dx) ** 2
    )
    if few_snaps:
        spread *= 1.35
    spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)

    quantiles = []
    for z in Z:
        q = pct_of_logit(median_logit + z * spread)
        q = min(max(q, 0.0), 100.0)
        quantiles.append(round(q, 4))
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def forecast_case(case):
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    target_hours = hours_since_epoch(parse_time(case["target"]))
    x_target = x_of_hours(target_hours)
    days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)

    snaps = snapshots(history)
    if not snaps:
        return [{"id": t["id"], "q": [0.01, 0.05, 0.1, 0.5, 2.0]} for t in trophies]

    ws = weights_for(snaps)
    players_hat = forecast_players(snaps, ws, x_target, days)
    few_snaps = len(snaps) < 4
    x_last = x_of_hours(snaps[-1]["hours"])
    dx = max(x_target - x_last, 0.0)

    rows = []
    for idx, trophy in enumerate(trophies):
        series = []
        for s in snaps:
            rate = s["rates"][idx] if idx < len(s["rates"]) else None
            if rate is not None:
                series.append((s["hours"], float(rate), s["players"]))
        if not series:
            rows.append({"id": trophy["id"], "q": [0.01, 0.02, 0.05, 0.2, 1.0]})
            continue
        q = forecast_trophy(series, x_target, dx, days, players_hat, few_snaps)
        rows.append({"id": trophy["id"], "q": q})
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        rows = forecast_case(case)
        out_path = OUTPUT_DIR / (path.name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"trophies": rows}, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
