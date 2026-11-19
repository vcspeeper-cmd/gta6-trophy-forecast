# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vcspeeper-cmd/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vcspeeper-cmd/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

The forecaster works on the log-odds scale. Each trophy's last observed unlock
rate is mapped through `logit`, then shifted by a pooled population term and a
per-trophy drift term. The population term fits `ln(players_tracked)` against
time over the recent window, projects it to the target minute, and multiplies
the resulting log-population change by a pooled slope `b` estimated from
consecutive distinct reads whose player counts differ. Drift comes from a
least-squares fit of each trophy's log-odds against time, damped by a factor of
0.25 so short histories cannot extrapolate wildly.

Uncertainty is a per-trophy sigma built from three parts: half the
root-mean-square residual of that trophy's own fit, an extrapolation term that
grows with the horizon relative to the observed span, and a small fixed floor.
Cases with fewer than two distinct reads get an extra horizon-dependent floor,
and sigma is always clamped to a sane range. The five requested quantiles are
then `expit(mu + z*sigma)` at the standard normal quantiles, clamped to
`[0.01, 99.99]`, forced non-decreasing, and rounded to six decimals. The program
reads `*.in` files from `MAKEFUN_INPUT_DIR` and writes one `<id>.out` per case
to `MAKEFUN_OUTPUT_DIR`, using only the standard library.
