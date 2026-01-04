#!/usr/bin/env python3
"""
QQQ 2-DTE wing-selection assistant (deterministic).

Data:
- Daily OHLC from Stooq CSV (date-ranged)
- Live-ish P_ref from Yahoo Finance quote JSON (pre/regular/mark fallback)

Outputs:
- Exactly the ordered report you specified
- Hard-fails if required data is missing or invalid (no guessing)
"""

from __future__ import annotations

import math
import sys
import json
import time
import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from typing import Dict, Tuple, List, Optional

import requests
import pandas as pd


LA = ZoneInfo("America/Los_Angeles")


# ---------- Config ----------
STOOQ_BASE = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=QQQ"

MIN_TRADING_ROWS = 60
CALENDAR_BUFFER_DAYS = 320
EXTRA_BUFFER_DAYS = 120  # used if initial range doesn't yield enough rows

GAP_SKIP_THRESHOLD = 0.0135  # 1.35%

# Coefficients (latest backtest)
COEFFS = {1: 3.0, 2: 3.0, 3: 3.0, 5: 3.0, 10: 2.5, 20: 1.5}

# Wing mapping
WING_MAP = {
    "Quiet": (0.8, 1.2),
    "Mid Bull": (1.0, 1.4),
    "Mid Bear": (1.4, 1.0),
    "High Bull": (1.2, 1.4),
    "High Bear": (1.4, 1.2),
    "High Both": (1.6, 1.6),
}


# ---------- Helpers ----------
def now_la() -> dt.datetime:
    return dt.datetime.now(tz=LA)

def yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def http_get(url: str, timeout: int = 20, retries: int = 2, backoff_s: float = 1.5) -> str:
    last_err = None
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; QQQWingBot/1.0; +https://example.com)"
    }
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff_s * (i + 1))
    raise RuntimeError(f"HTTP GET failed: {url} :: {last_err}")

def validate_stooq_csv(text: str) -> None:
    if not text or "Date,Open,High,Low,Close,Volume" not in text.splitlines()[0]:
        raise RuntimeError("Stooq CSV validation failed: missing header")
    # Must have at least a few rows beyond header
    if len(text.splitlines()) < 10:
        raise RuntimeError("Stooq CSV validation failed: too few rows (likely empty/blocked)")

def fetch_stooq_ohlc(today: dt.date) -> pd.DataFrame:
    """
    Fetch date-ranged daily OHLC from Stooq.
    If not enough rows, extend range earlier and retry.
    """
    d2 = today
    d1 = today - dt.timedelta(days=CALENDAR_BUFFER_DAYS)

    for attempt in range(2):
        url = f"{STOOQ_BASE}&d1={yyyymmdd(d1)}&d2={yyyymmdd(d2)}"
        csv_text = http_get(url)
        validate_stooq_csv(csv_text)

        df = pd.read_csv(pd.compat.StringIO(csv_text))
        # normalize
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.sort_values("Date").reset_index(drop=True)

        # basic sanity: required columns
        for c in ["Open", "High", "Low", "Close"]:
            if c not in df.columns:
                raise RuntimeError(f"Stooq CSV parse failed: missing column {c}")

        # If enough completed sessions exist, return
        if len(df) >= MIN_TRADING_ROWS:
            return df

        # else expand earlier
        d1 = d1 - dt.timedelta(days=EXTRA_BUFFER_DAYS)

    raise RuntimeError(f"Stooq history insufficient: got {len(df)} rows, need >= {MIN_TRADING_ROWS}")

def find_d0(df: pd.DataFrame, today: dt.date) -> dt.date:
    """
    D0 = last completed trading day (Date < today).
    """
    candidates = df[df["Date"] < today]
    if candidates.empty:
        raise RuntimeError("Cannot determine D0: no row with Date < today")
    return candidates.iloc[-1]["Date"]

def get_live_pref_price() -> Tuple[float, str, Dict]:
    """
    Get P_ref from Yahoo quote JSON.
    Prefer preMarketPrice if present, else regularMarketPrice.
    """
    raw = http_get(YAHOO_QUOTE_URL)
    data = json.loads(raw)
    res = data.get("quoteResponse", {}).get("result", [])
    if not res:
        raise RuntimeError("Yahoo quote failed: empty result")
    q = res[0]

    # Try best available fields
    for field in ["preMarketPrice", "regularMarketPrice", "postMarketPrice"]:
        v = q.get(field)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), field, q

    raise RuntimeError("Yahoo quote failed: no usable price field")

def compute_sL(df: pd.DataFrame, d0: dt.date, L: int) -> Tuple[int, str]:
    """
    Compute sL using windows ending at D0.
    """
    # Locate D0 index
    idx = df.index[df["Date"] == d0]
    if len(idx) == 0:
        raise RuntimeError(f"D0 {d0} not found in df")
    i0 = int(idx[0])

    # Need 2L days ending at i0
    start_curr = i0 - (L - 1)
    start_prev = i0 - (2 * L - 1)
    end_prev = i0 - L

    if start_prev < 0:
        raise RuntimeError(f"Insufficient history to compute L={L} at D0={d0}")

    curr = df.iloc[start_curr : i0 + 1]
    prev = df.iloc[start_prev : end_prev + 1]

    H_curr = float(curr["High"].max())
    L_curr = float(curr["Low"].min())
    C_curr = float(df.iloc[i0]["Close"])
    H_prev = float(prev["High"].max())
    L_prev = float(prev["Low"].min())

    # Rules
    if (H_curr > H_prev) and (C_curr > H_prev):
        return +2, "Acceptance Up"
    elif (L_curr < L_prev) and (C_curr < L_prev):
        return -2, "Acceptance Down"
    elif (L_curr < L_prev) and (C_curr > L_prev):
        return +1, "Failed Sweep Down"
    elif (H_curr > H_prev) and (C_curr < H_prev):
        return -1, "Failed Sweep Up"
    else:
        return 0, "Neutral"

def bucket_from_S(S: float) -> str:
    A = abs(S)
    if A < 16:
        return "Quiet"
    if 16 <= A < 22:
        if S > 0: return "Mid Bull"
        if S < 0: return "Mid Bear"
        return "Quiet"
    if 22 <= A < 32:
        if S > 0: return "High Bull"
        if S < 0: return "High Bear"
        return "Quiet"
    return "High Both"

def strikes_from_wings(P_ref: float, put_w: float, call_w: float) -> Tuple[int, int]:
    put_boundary = P_ref * (1 - put_w / 100.0)
    call_boundary = P_ref * (1 + call_w / 100.0)
    return math.floor(put_boundary), math.ceil(call_boundary)


# ---------- Main ----------
def main() -> int:
    t = now_la()
    today = t.date()

    # 0) Trading day check (simple): if weekend -> closed.
    # (If you want full holiday accuracy, add pandas_market_calendars.)
    if today.weekday() >= 5:
        print("GRAY — Market Closed")
        return 0

    # 1) Stooq OHLC
    df = fetch_stooq_ohlc(today)
    d0 = find_d0(df, today)

    # Require at least 60 sessions ending at D0
    df_upto_d0 = df[df["Date"] <= d0].copy()
    if len(df_upto_d0) < MIN_TRADING_ROWS:
        raise RuntimeError(f"Insufficient sessions ending at D0={d0}: {len(df_upto_d0)} < {MIN_TRADING_ROWS}")

    close_prev = float(df_upto_d0.iloc[-1]["Close"])

    # 2) P_ref
    P_ref, price_field, yahoo_blob = get_live_pref_price()

    # Optional O_today from Yahoo if present (context only)
    O_today = yahoo_blob.get("regularMarketOpen")
    O_today = float(O_today) if isinstance(O_today, (int, float)) else None

    # 3) Gap skip (P_ref vs Close_prev)
    gap = abs(P_ref - close_prev) / close_prev

    if gap > GAP_SKIP_THRESHOLD:
        # Still must print all intermediate numbers you need (within the "SKIP" path)
        ts = t.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Action: SKIP")
        print(f"Timestamp (PDT) and P_ref, Close_prev, gap%:")
        print(f"{ts} | P_ref={P_ref:.2f} ({price_field}) | Close_prev={close_prev:.2f} | gap%={gap*100:.3f}%")
        print("Signals table: (not computed due to SKIP)")
        print("Score: (not computed due to SKIP)")
        print("Bucket: (not computed due to SKIP)")
        print("Chosen wings: 1.6% / 1.6% (fallback if you insist; size down)")
        putS, callS = strikes_from_wings(P_ref, 1.6, 1.6)
        print(f"Strikes to place today: PutStrike={putS} / CallStrike={callS}")
        print("Static comparisons:")
        a_put, a_call = strikes_from_wings(P_ref, 0.8, 1.2)
        b_put, b_call = strikes_from_wings(P_ref, 1.0, 1.0)
        print(f"0.8/1.2 strikes: {a_put} / {a_call}")
        print(f"1.0/1.0 strikes: {b_put} / {b_call}")
        print("Risk notes:")
        print("- size down; gap regime")
        return 0

    # 4) Signals
    signals: Dict[int, Tuple[int, str]] = {}
    for L in [1, 2, 3, 5, 10, 20]:
        signals[L] = compute_sL(df_upto_d0, d0, L)

    s1, s2, s3, s5, s10, s20 = (signals[1][0], signals[2][0], signals[3][0], signals[5][0], signals[10][0], signals[20][0])

    # 5) Score
    S = (
        COEFFS[1]*s1 + COEFFS[2]*s2 + COEFFS[3]*s3 +
        COEFFS[5]*s5 + COEFFS[10]*s10 + COEFFS[20]*s20
    )
    A = abs(S)

    # 6) Bucket
    bucket = bucket_from_S(S)

    # 7) Wings
    put_w, call_w = WING_MAP[bucket]

    # 8) Strikes
    put_strike, call_strike = strikes_from_wings(P_ref, put_w, call_w)

    # 9) Static benchmarks
    staticA = strikes_from_wings(P_ref, 0.8, 1.2)
    staticB = strikes_from_wings(P_ref, 1.0, 1.0)

    # 10) Final report (exact order)
    ts = t.strftime("%Y-%m-%d %H:%M:%S")

    print("Action: TRADE")
    print(f"Timestamp (PDT) and P_ref, Close_prev, gap%:")
    o_today_str = f"{O_today:.2f}" if O_today is not None else "N/A"
    print(f"{ts} | P_ref={P_ref:.2f} ({price_field}) | Close_prev={close_prev:.2f} | gap%={gap*100:.3f}% | O_today={o_today_str}")

    print("Signals table: s1,s2,s3,s5,s10,s20 with labels")
    for L in [1,2,3,5,10,20]:
        s, lbl = signals[L]
        print(f"L={L}: s{L}={s:+d} ({lbl})")

    print("Score: S line + A=|S|")
    print(f"S = 3*({s1}) + 3*({s2}) + 3*({s3}) + 3*({s5}) + 2.5*({s10}) + 1.5*({s20}) = {S:.1f}")
    print(f"A = |S| = {A:.1f}")

    print(f"Bucket: {bucket}")

    print(f"Chosen wings: PutWing% / CallWing% + one-line reason")
    print(f"{put_w:.1f}% / {call_w:.1f}% — Bucket={bucket} because |S|={A:.1f}; wing={put_w:.1f}/{call_w:.1f} by mapping rule.")

    print(f"Strikes to place today: PutStrike / CallStrike (based on P_ref)")
    print(f"{put_strike} / {call_strike}")

    print("Static comparisons:")
    print(f"0.8/1.2 strikes: {staticA[0]} / {staticA[1]}")
    print(f"1.0/1.0 strikes: {staticB[0]} / {staticB[1]}")

    print("Risk notes (max 3 bullets):")
    if "Bear" in bucket:
        print("- put-side risk elevated")
    if "Bull" in bucket:
        print("- call-side risk elevated")
    if bucket == "High Both":
        print("- size down; expect chop")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        # Exact failure reporting, no guessing
        print(f"FAILED — {e}")
        raise
