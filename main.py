"""Launch-day Twitch viewer forecaster (make.fun: gta6-twitch-ccu-forecast-01).

Reads one case (everything known up to the end of hour 1 of a game's release
day, UTC) and writes, for each of hours 2..24, five quantiles (5/25/50/75/95%)
of that hour's mean concurrent viewers in the game's Twitch category.

Method, in one paragraph: the case file carries the full hourly history of
dozens of past big launches ("comparables"). Launches whose first hour was
already a real, post-unlock hour (a rolling-midnight console launch looks
like this; a single-global-unlock launch does not, and its hour 1 is noise)
form a pool. For every later hour, a small ridge regression predicts the
hour's log viewer count from four features of the case at the checkpoint --
the log of hour 1's count, how deep hour 1 sits below the pre-release
evening's peak (the "trough depth": for a rolling unlock the evening before
UTC midnight is already playing, and hour 1 lands in the lull after it), the
log viewers-per-channel of hour 1, and the momentum from the last
pre-release hour into hour 1 -- shrunk toward the pool's median log-ratio
for that hour. The five quantiles come from the pool's leave-one-out
residuals of that same fit, slightly widened, and the median curve is
lightly smoothed across neighbouring hours. Everything is computed from the
case file at run time: no network, no randomness, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantiles at LEVELS, for the fallback forecast.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

FEATS = ("a1", "g", "vpc", "mom")
LAM = 2.0          # ridge strength shrinking slopes to the ratio model
SPREAD = 1.05      # widening applied to the residual quantiles
SMOOTH = 0.5       # pull of each hour's median toward its neighbours' mean
MIN_FRAC = 0.12    # a pool launch's hour 1 must reach this share of its day peak
MIN_ABS = 2000.0   # ... and at least this many viewers
MIN_POOL = 8       # below this many pool launches, use the fallback
LOG_FALLBACK_VPC = math.log(50.0)
TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_key(moment):
    return moment.strftime(TIME_FMT)


def make_features(first_viewers, first_channels, pre_values):
    """The four checkpoint features, from hour 1 and the hours just before it.

    pre_values are the up-to-8 hourly viewer counts right before the day
    starts, oldest first, with None where an hour is missing.
    """
    a1 = math.log(max(1.0, float(first_viewers)))
    present = [v for v in pre_values if v is not None and v > 0]
    pre_peak = max(present) if present else None
    g = math.log(pre_peak) - a1 if pre_peak else 0.0
    last_pre = None
    for v in reversed(pre_values):
        if v is not None and v > 0:
            last_pre = v
            break
    mom = math.log(max(1.0, float(first_viewers)) / last_pre) if last_pre else 0.0
    if first_channels:
        vpc = math.log(max(1.0, float(first_viewers)) / float(first_channels))
    else:
        vpc = LOG_FALLBACK_VPC
    return {"a1": a1, "g": g, "vpc": vpc, "mom": mom}


def day_windows(comp):
    """Split one comparable's hourly points into (day, day_channels, pre).

    day is the 24 hourly viewer counts of its release day, day_channels the
    matching channel counts, pre the 8 hourly counts before the day starts;
    each entry is None where the source has no point for that hour.
    """
    start = parse_time(comp["day_start"])
    viewers = {}
    channels = {}
    for pt in comp.get("hourly") or []:
        viewers[pt["from"]] = pt.get("viewers")
        channels[pt["from"]] = pt.get("channels")
    day = [viewers.get(hour_key(start + timedelta(hours=h))) for h in range(24)]
    day_ch = [channels.get(hour_key(start + timedelta(hours=h))) for h in range(24)]
    pre = [viewers.get(hour_key(start - timedelta(hours=k))) for k in range(8, 0, -1)]
    return day, day_ch, pre


def build_pool(case):
    """Comparable launches usable as shape evidence for this case."""
    pool = []
    for comp in case.get("comparables") or []:
        unlock = comp.get("unlock") or ""
        if "Not a launch" in unlock or "before release" in (comp.get("name") or ""):
            continue  # the case game's own pre-release category history
        day, day_ch, pre = day_windows(comp)
        present = [v for v in day if v is not None]
        if len(present) < 20 or day[0] is None:
            continue
        if day[0] < MIN_ABS or day[0] < MIN_FRAC * max(present):
            continue  # hour 1 was pre-unlock noise for this launch
        if any(v is None or v <= 0 for v in day):
            continue
        feats = make_features(day[0], day_ch[0], pre)
        pool.append({"feats": feats, "day": day})
    return pool


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gauss-Jordan elimination."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        diag = rows[col][col]
        for r in range(n):
            if r != col:
                factor = rows[r][col] / diag
                for c in range(col, n + 1):
                    rows[r][c] -= factor * rows[col][c]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def fit_hour(pool, hour, lam):
    """Ridge fit of one hour's log count on the checkpoint features.

    The fit is done on z = log(viewers) - a1 - median_ratio, so zero slopes
    recover the plain ratio model, and the ridge shrinks toward it.
    Returns (predict, loo_residuals): the fit's predictor, and each pool
    member's leave-one-out residual, for honest quantiles.
    """
    ys = [math.log(e["day"][hour - 1]) for e in pool]
    n = len(pool)
    ratios = sorted(ys[i] - pool[i]["feats"]["a1"] for i in range(n))
    med_ratio = ratios[n // 2]
    means = {f: sum(e["feats"][f] for e in pool) / n for f in FEATS}
    dim = len(FEATS)

    def fit(sub):
        m = len(sub)
        sub_ys = [math.log(e["day"][hour - 1]) for e in sub]
        sub_ratios = sorted(sub_ys[i] - sub[i]["feats"]["a1"] for i in range(m))
        sub_med = sub_ratios[m // 2]
        sub_means = {f: sum(e["feats"][f] for e in sub) / m for f in FEATS}
        xtx = [[0.0] * dim for _ in range(dim)]
        xtz = [0.0] * dim
        for k, e in enumerate(sub):
            x = [e["feats"][f] - sub_means[f] for f in FEATS]
            z = sub_ys[k] - e["feats"]["a1"] - sub_med
            for i in range(dim):
                xtz[i] += x[i] * z
                for j in range(dim):
                    xtx[i][j] += x[i] * x[j]
        for i in range(dim):
            xtx[i][i] += lam
        slopes = solve_linear(xtx, xtz)

        def predict(feats):
            return (feats["a1"] + sub_med
                    + sum(slopes[i] * (feats[f] - sub_means[f]) for i, f in enumerate(FEATS)))

        return predict

    predict = fit(pool)
    loo = []
    for k in range(n):
        sub = pool[:k] + pool[k + 1:]
        loo.append(ys[k] - fit(sub)(pool[k]["feats"]))
    return predict, sorted(loo)


def interp_quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] * (1 - (pos - lo)) + sorted_vals[hi] * (pos - lo)


def model_forecast(case, first_viewers, first_channels):
    """The ridge/pool forecast: {hour: [5 log-quantiles]} or None."""
    day0 = parse_time(case["day_start"])
    hourly = {row["from"]: row.get("viewers") for row in case.get("history_hourly") or []}
    pre = [hourly.get(hour_key(day0 - timedelta(hours=k))) for k in range(8, 0, -1)]
    target = make_features(first_viewers, first_channels, pre)
    pool = build_pool(case)
    if len(pool) < MIN_POOL:
        return None
    raw = {}
    for row in case["forecast"]:
        h = row["hour"]
        predict, loo = fit_hour(pool, h, LAM)
        centre = predict(target)
        qs = [interp_quantile(loo, t) for t in LEVELS]
        med = qs[2]
        raw[h] = [centre + med + (q - med) * SPREAD for q in qs]
    if SMOOTH > 0:
        meds = {h: raw[h][2] for h in raw}
        for h in raw:
            near = [meds[k] for k in (h - 1, h, h + 1) if k in meds]
            shift = (sum(near) / len(near) - meds[h]) * SMOOTH
            raw[h] = [v + shift for v in raw[h]]
    return raw


def fallback_forecast(case, p1):
    """The house baseline's shape, used only if the pool is unusable."""
    raw = {}
    for row in case["forecast"]:
        h = row["hour"]
        raw[h] = [math.log(p1) + z * 0.15 * math.sqrt(h - 1) for z in Z]
    return raw


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def forecast_case(case):
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    try:
        raw = model_forecast(case, first["viewers"], first.get("channels"))
    except Exception:
        raw = None
    if raw is None:
        raw = fallback_forecast(case, p1)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = []
        prev = 0.0
        for v in raw[h]:
            q = math.exp(v)
            q = min(max(q, 0.01), 1e9)
            q = max(q, prev)  # quantiles never decrease
            qs.append(round2(q))
            prev = q
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
