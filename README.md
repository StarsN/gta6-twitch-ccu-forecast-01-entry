[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/StarsN/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/StarsN/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewer forecast

Entry for the make.fun competition **gta6-twitch-ccu-forecast-01**: given a
case file with everything recorded up to the end of hour 1 of a game's
release day (UTC), forecast the 5/25/50/75/95% quantiles of the mean Twitch
category viewer count for each of hours 2–24.

## How it works

`main.py` (Python 3, standard library only) reads `/input/<id>.in` and writes
`/output/<id>.out`. Everything is derived from the case file at run time —
no network, no randomness, deterministic output.

1. **Pool.** From the case's past-launch comparables, keep the launches whose
   hour 1 was already a genuine post-unlock hour (rolling-midnight style
   launches): hour 1 must reach at least 12% of that launch's day peak and at
   least 2,000 viewers. The case game's own pre-release category history is
   excluded.
2. **Features.** At the checkpoint: the log of hour 1's viewers; the "trough
   depth" — how far hour 1 sits below the pre-release evening's peak (for a
   rolling unlock the evening before UTC midnight is already playing, and
   hour 1 lands in the lull after it); the log viewers-per-channel of hour 1;
   and the momentum from the last pre-release hour into hour 1.
3. **Per-hour ridge regression** of each later hour's log viewers on those
   features, shrunk toward the pool's median log-ratio for that hour (the
   pure ratio model is the prior). Quantiles come from the pool's
   leave-one-out residuals of the same fit, slightly widened; the median
   curve is lightly smoothed across neighbouring hours.

## Local run

```
MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=./output python3 main.py
```
