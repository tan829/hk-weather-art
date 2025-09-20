#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_hk_weather.py — 抓取香港逐小时天气数据并清洗为 visualize_weather.py 可用的 CSV

输出列：
  time, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m
"""

import argparse
import sys
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
import pandas as pd
import requests

HK_LAT, HK_LON = 22.2793, 114.1628
TZ = "Asia/Hong_Kong"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
HOURLY_STR = ",".join(HOURLY_VARS)

def _fetch(url, params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def _json_to_df(j):
    hourly = (j or {}).get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=["time"] + HOURLY_VARS)
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for col in HOURLY_VARS:
        vals = hourly.get(col, None)
        df[col] = pd.to_numeric(vals, errors="coerce") if vals is not None else None
    return df

def fetch_recent_days(recent_days: int, tz: str = TZ) -> pd.DataFrame:
    """把最近 N 天拆分为：历史（到昨天，用 ERA5） + 今天（用 forecast）"""
    today = date.today()
    start_date = today - timedelta(days=recent_days-1)
    yesterday = today - timedelta(days=1)

    dfs = []

    # 历史部分：如果跨度包含昨天及更早，走 ERA5
    if start_date <= yesterday:
        params = dict(
            latitude=HK_LAT, longitude=HK_LON,
            hourly=HOURLY_STR, timezone=tz,
            start_date=start_date.isoformat(),
            end_date=yesterday.isoformat(),
        )
        js = _fetch(ARCHIVE_URL, params)
        df_a = _json_to_df(js)
        dfs.append(df_a)

    # 今天起（含未来）：走 forecast
    params2 = dict(
        latitude=HK_LAT, longitude=HK_LON,
        hourly=HOURLY_STR, timezone=tz,
        start_date=today.isoformat(),
        end_date=today.isoformat(),   # 只取今天，避免把未来混进来（你要未来也行：去掉 end_date）
    )
    js2 = _fetch(FORECAST_URL, params2)
    df_f = _json_to_df(js2)
    dfs.append(df_f)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["time"]+HOURLY_VARS)
    return df

def fetch_range(start_date: str, end_date: str, tz: str = TZ) -> pd.DataFrame:
    """明确时间段：过去走 ERA5；含今天/未来则拆分后用 forecast 合并"""
    start = datetime.fromisoformat(start_date).date()
    end   = datetime.fromisoformat(end_date).date()
    if end < start:
        raise ValueError("END 不能早于 START")

    today = date.today()
    dfs = []

    # 过去部分（到昨天）：ERA5
    past_end = min(end, today - timedelta(days=1))
    if start <= past_end:
        params = dict(
            latitude=HK_LAT, longitude=HK_LON,
            hourly=HOURLY_STR, timezone=tz,
            start_date=start.isoformat(), end_date=past_end.isoformat(),
        )
        js = _fetch(ARCHIVE_URL, params)
        dfs.append(_json_to_df(js))

    # 今天及以后（若 end 覆盖到今天）：forecast
    if end >= today:
        params2 = dict(
            latitude=HK_LAT, longitude=HK_LON,
            hourly=HOURLY_STR, timezone=tz,
            start_date=today.isoformat(), end_date=today.isoformat(),
        )
        js2 = _fetch(FORECAST_URL, params2)
        dfs.append(_json_to_df(js2))

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["time"]+HOURLY_VARS)
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # 基础检查
    if df.empty or "time" not in df:
        raise RuntimeError("API 返回为空（没有 time）。请稍后重试或改用 --range 指定历史日期。")

    # 排序、去重
    df = df.sort_values("time").drop_duplicates("time")

    # 统一到整点小时
    df = (df.set_index("time").resample("1H").agg({
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "precipitation": "sum",
        "wind_speed_10m": "mean",
    }).reset_index())

    # 填充与插值
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0)
    for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").interpolate(limit_direction="both")

    # 如果还是存在整列缺失，直接报错，避免生成“空白 CSV”
    missing_cols = [c for c in HOURLY_VARS if df[c].isna().all()]
    if missing_cols:
        raise RuntimeError(f"以下列未从 API 获得有效数据：{missing_cols}。"
                           f"请改用 --range 指向过去日期（例如昨天之前），或稍后再试。")

    return df

def parse_args():
    p = argparse.ArgumentParser(description="Fetch Hong Kong hourly weather to CSV for visualization")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--recent-days", type=int, help="抓取最近 N 天（包含今天）")
    g.add_argument("--range", nargs=2, metavar=("START", "END"),
                   help="抓取指定日期区间（YYYY-MM-DD YYYY-MM-DD）")
    p.add_argument("--out", default="data/hk_weather.csv", help="输出 CSV 路径")
    p.add_argument("--tz", default=TZ, help="时区（默认 Asia/Hong_Kong）")
    return p.parse_args()

def main():
    args = parse_args()
    if args.range:
        start, end = args.range
        raw = fetch_range(start, end, tz=args.tz)
    else:
        raw = fetch_recent_days(args.recent_days, tz=args.tz)

    df = clean_df(raw)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    t0, t1 = df["time"].min(), df["time"].max()
    print(f"Saved CSV: {args.out}  ({len(df)} rows)  Range: {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
