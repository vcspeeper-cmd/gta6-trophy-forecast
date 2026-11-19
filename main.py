#!/usr/bin/env python3
"""Forecast GTA6 trophy rarity quantiles from tracked-player history.

Reads every ``*.in`` case in the input directory and writes one ``<id>.out``
per case.  Standard library only, deterministic, no network.
"""

import glob
import json
import math
import os

# Clamp bounds for percentages.
PCT_LO = 0.01
PCT_HI = 99.99

SCHEMA = "makefun/gta6-trophies-forecast/v1"

# Standard normal quantiles for the five requested levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

EPS = 1e-12


def logit(p):
    """Map a percent to the log-odds scale."""
    if p < PCT_LO:
        p = PCT_LO
    elif p > PCT_HI:
        p = PCT_HI
    return math.log(p / (100.0 - p))


def expit(x):
    """Inverse of :func:`logit`; returns a percent, saturating far out."""
    if x > 40.0:
        return PCT_HI
    if x < -40.0:
        return PCT_LO
    return 100.0 / (1.0 + math.exp(-x))


def minutes(iso):
    """ISO-8601 UTC stamp (``...Z``) -> minutes since the Unix epoch."""
    text = iso.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1]
    if "T" in text:
        date_part, clock_part = text.split("T", 1)
    elif " " in text:
        date_part, clock_part = text.split(" ", 1)
    else:
        date_part, clock_part = text, "00:00:00"

    y, m, d = (int(part) for part in date_part.split("-"))

    clock_part = clock_part.split("+")[0].split("-")[0]
    pieces = clock_part.split(":")
    hh = int(pieces[0]) if len(pieces) > 0 and pieces[0] else 0
    mm = int(pieces[1]) if len(pieces) > 1 and pieces[1] else 0
    ss = float(pieces[2]) if len(pieces) > 2 and pieces[2] else 0.0

    # Civil-from-days (Howard Hinnant's algorithm).
    yy = y - (1 if m <= 2 else 0)
    era = (yy if yy >= 0 else yy - 399) // 400
    yoe = yy - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468

    return days * 1440.0 + hh * 60.0 + mm + ss / 60.0


def ols(xs, ys):
    """Ordinary least squares fit; returns (slope, intercept, rms_residual)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, ys[0], 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < EPS:
        slope = 0.0
        intercept = sy / n
    else:
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
    resid = 0.0
    for x, y in zip(xs, ys):
        diff = y - (intercept + slope * x)
        resid += diff * diff
    return slope, intercept, math.sqrt(resid / n)


def safe_log_players(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value < 1.0:
        value = 1.0
    return math.log(value)


def load_reads(case, n_trophies):
    """Step 3: keep usable history reads as (minutes, players, rates)."""
    history = case.get("history") or []
    kept = []
    for read in history:
        rates = read.get("rates")
        if not isinstance(rates, list) or len(rates) < n_trophies:
            continue
        if any(rate is None for rate in rates):
            continue
        try:
            t = minutes(read["at"])
            players = float(read.get("players_tracked") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        kept.append((t, players, [float(r) for r in rates[:n_trophies]]))

    if kept:
        return kept

    # Nothing survived: pad the rates with 0.5 instead of dropping the read.
    for read in history:
        rates = read.get("rates")
        if not isinstance(rates, list):
            rates = []
        padded = []
        for i in range(n_trophies):
            value = rates[i] if i < len(rates) else None
            padded.append(0.5 if value is None else float(value))
        try:
            t = minutes(read["at"])
            players = float(read.get("players_tracked") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        kept.append((t, players, padded))
    return kept


def distinct_reads(reads):
    """Step 6: collapse runs repeating both players count and every rate."""
    out = []
    for read in reads:
        if out:
            prev = out[-1]
            if prev[1] == read[1] and prev[2] == read[2]:
                continue
        out.append(read)
    return out


def forecast(case):
    trophies = case.get("trophies") or []
    n = len(trophies)
    if n == 0:
        return {"trophies": []}

    reads = load_reads(case, n)

    if reads:
        t_last, p_last, r_last = reads[-1]
        t_first = reads[0][0]
    else:
        # No history at all: fall back to a flat 0.5% prior.
        t_last = minutes(case.get("at", "1970-01-01T00:00:00Z"))
        t_first = t_last
        p_last = 1.0
        r_last = [0.5] * n

    try:
        t_target = minutes(case["target"])
    except (KeyError, TypeError, ValueError):
        t_target = t_last

    horizon = t_target - t_last
    if horizon < 1.0:
        horizon = 1.0
    span = t_last - t_first
    if span < 1.0:
        span = 1.0

    # Step 5: predict the tracked population at the target time.
    window_start = t_last - max(0.6 * span, 30.0)
    recent = [read for read in reads if read[0] >= window_start]
    if not recent:
        recent = reads[-1:] if reads else []
    log_p_target = math.log(max(p_last, 1.0))
    if recent:
        xs = [read[0] for read in recent]
        ys = [safe_log_players(read[1]) for read in recent]
        slope, intercept, _ = ols(xs, ys)
        if slope < 0.0:
            slope = 0.0
        if len(recent) == 1:
            fitted = ys[0]
        else:
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            fitted = mean_y + slope * (t_target - mean_x)
        log_p_target = fitted
    floor = math.log(p_last) if p_last > 0.0 else 0.0
    if log_p_target < floor:
        log_p_target = floor
    dlogp = log_p_target - math.log(max(p_last, 1.0))

    # Step 6: distinct reads.
    distinct = distinct_reads(reads)

    # Step 7: pooled population slope b.
    sum_dxdy = 0.0
    sum_dxdx = 0.0
    for a, b_read in zip(distinct, distinct[1:]):
        pa = a[1] if a[1] > 0.0 else 1.0
        pb = b_read[1] if b_read[1] > 0.0 else 1.0
        if pa == pb:
            continue
        dx = math.log(pb) - math.log(pa)
        sum_dxdx += dx * dx
        for i in range(n):
            dy = logit(b_read[2][i]) - logit(a[2][i])
            sum_dxdy += dx * dy
    if sum_dxdx < EPS:
        b = 0.0
    else:
        b = sum_dxdy / sum_dxdx
    if b < -1.5:
        b = -1.5
    elif b > 0.0:
        b = 0.0

    # Steps 8 and 9: per-trophy drift and spread.
    extrap = 0.10 * math.log(1.0 + horizon / max(span, 60.0))
    enough = len(distinct) >= 3
    d_times = [read[0] for read in distinct]

    drift = [0.0] * n
    sigma = [0.0] * n
    for i in range(n):
        rms = 0.0
        if enough:
            ys = [logit(read[2][i]) for read in distinct]
            slope, _intercept, rms = ols(d_times, ys)
            drift[i] = slope * horizon * 0.25
        s = math.sqrt((0.5 * rms) ** 2 + extrap ** 2 + 0.04 ** 2)
        if len(distinct) < 2:
            floor_sigma = 0.45 + 0.15 * math.log(1.0 + horizon / 60.0)
            if s < floor_sigma:
                s = floor_sigma
        if s < 0.04:
            s = 0.04
        elif s > 0.85:
            s = 0.85
        sigma[i] = s

    # Step 10: quantiles.
    rows = []
    for i, trophy in enumerate(trophies):
        mu = logit(r_last[i]) + b * dlogp + drift[i]
        q = []
        for z in Z:
            value = expit(mu + z * sigma[i])
            if value < PCT_LO:
                value = PCT_LO
            elif value > PCT_HI:
                value = PCT_HI
            q.append(value)
        for j in range(1, len(q)):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        q = [round(value, 6) for value in q]
        rows.append({"id": trophy.get("id"), "q": q})

    return {"trophies": rows}


def main():
    indir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(outdir, exist_ok=True)

    written = 0
    for path in sorted(glob.glob(os.path.join(indir, "*.in"))):
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast(case)
        case_id = case.get("id")
        if case_id is None:
            case_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(outdir, "%s.out" % case_id)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))
        written += 1
    print("wrote %d cases" % written)


if __name__ == "__main__":
    main()
