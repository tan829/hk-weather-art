#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_weather.py — Last 10 Days · No-Parameter · Final Fix Version
Reads:  data/hk_weather.csv
Outputs:  art/hk_weather_poster.png
Style:  Dark background + Temperature gradient line (with color bar) + Humidity ribbon + Precipitation bars + Wind speed neon line
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.dates import DateFormatter, AutoDateLocator

# ========= Constants =========
CSV_PATH = Path("data/hk_weather.csv")
OUT_PATH = Path("art/hk_weather_poster.png")
SMOOTH_WIN = 3
TITLE = "Hong Kong Last 10 Days Hourly Weather Poster"
SUBTITLE = "Data: Open-Meteo (ERA5 / Forecast) | Last 10 Days"
POSTER_SIZE = (12, 16)
DPI = 160

# ========= Style and Fonts =========
def setup_style():
    for f in ["Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei"]:
        try:
            mpl.rcParams["font.sans-serif"] = [f]
            break
        except Exception:
            pass
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.unicode_minus"] = False
    bg, fg, grid = "#0b0f14", "#e6e6e6", "#2a3440"
    mpl.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "text.color": fg,
        "axes.labelcolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "axes.grid": True,
        "grid.color": grid,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "axes.edgecolor": grid,
    })
    return bg, fg

# ========= Helper Functions =========
def rolling(series: pd.Series, win: int):
    return series.rolling(win, min_periods=1, center=True).mean() if win and win > 1 else series

def make_cmap(colors):
    return LinearSegmentedColormap.from_list("temp_cmap", colors)

def valid_mask(x, y):
    return np.isfinite(mpl.dates.date2num(x)) & np.isfinite(y)

def masked_xy(x, y):
    """Returns: Mask m, masked x (values), masked y — for use with fill_between"""
    xnum = mpl.dates.date2num(x)
    yv = np.asarray(y, dtype=float)
    m = np.isfinite(xnum) & np.isfinite(yv)
    x_ma = np.ma.masked_array(xnum, mask=~m)
    y_ma = np.ma.masked_array(yv, mask=~m)
    return m, x_ma, y_ma

def gradient_line(ax, x, y, cmap, lw=2.6, z=3):
    m = valid_mask(x, y)
    xnum = mpl.dates.date2num(x[m])
    yv = y[m]
    if len(yv) < 2:
        return None
    pts = np.array([xnum, yv]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(np.nanmin(yv), np.nanmax(yv)))
    lc.set_array(yv)
    lc.set_linewidth(lw)
    lc.set_zorder(z)
    ax.add_collection(lc)
    ax.set_xlim(xnum.min(), xnum.max())
    return lc

def glow_line(ax, x, y, color="#f8d66d", repeats=10, alpha=0.12, lw=1.8, z=3):
    m = valid_mask(x, y)
    xv, yv = x[m], y[m]
    if len(yv) < 2:
        return False
    r, g, b, _ = to_rgba(color, 1.0)
    for i in range(repeats, 0, -1):
        ax.plot(xv, yv, linewidth=lw + i * 0.9, color=(r, g, b, alpha * (i / repeats)), zorder=z - 1)
    ax.plot(xv, yv, linewidth=lw, color=color, zorder=z)
    return True

# ========= Main Workflow =========
def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH.resolve()} Please run fetch_hk_weather.py to generate it.")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Read and force to numeric
    df = pd.read_csv(CSV_PATH, parse_dates=["time"])
    for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply rolling average
    for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]:
        if c in df:
            df[c] = rolling(df[c], SMOOTH_WIN)

    # Use only the last 10 days (approximately 240 hours)
    df = df.tail(240)

    t = df["time"]
    temp = df.get("temperature_2m", pd.Series(dtype=float))
    rh = df.get("relative_humidity_2m", pd.Series(dtype=float)).clip(0, 100)
    pr = df.get("precipitation", pd.Series(dtype=float)).clip(lower=0)
    ws = df.get("wind_speed_10m", pd.Series(dtype=float)).clip(lower=0)

    fig = plt.figure(figsize=POSTER_SIZE, dpi=DPI)
    ax_top = plt.subplot2grid((14, 1), (0, 0), rowspan=5)
    ax_mid = plt.subplot2grid((14, 1), (5, 0), rowspan=4)
    ax_bot = plt.subplot2grid((14, 1), (9, 0), rowspan=5)

    # Top: Temperature (gradient line + masked baseline fill)
    temp_cmap = make_cmap(["#2b90e9", "#73d2f6", "#ffd166", "#ff7c43", "#ef476f"])
    lc = None
    if "temperature_2m" in df.columns and len(temp) > 0:
        lc = gradient_line(ax_top, t, temp.values, temp_cmap, lw=2.6, z=3)
    m_t, t_ma_top, temp_ma = masked_xy(t, temp.values)
    if lc is not None and m_t.sum() >= 2:
        base_val = float(np.nanmin(temp[m_t]))
        base_ma = np.ma.masked_array(np.full_like(temp_ma, base_val, dtype=float), mask=temp_ma.mask)
        ax_top.fill_between(
            t_ma_top, temp_ma, base_ma, where=~t_ma_top.mask, color=to_rgba("#ef476f", 0.20), zorder=1
        )
        ax_top.set_ylabel("Temperature (°C)")
        ax_top.set_ylim(base_val - 1.5, float(np.nanmax(temp[m_t])) + 1.5)
        cbar = fig.colorbar(lc, ax=ax_top, pad=0.01, fraction=0.03)
        cbar.set_label("Temperature (°C)")
        ax_top.set_title(TITLE, loc="left", pad=16, fontsize=18, fontweight=600)
    else:
        ax_top.text(0.02, 0.85, "Temperature: No data available", transform=ax_top.transAxes, color="#a8b3c3")

    # Middle: Humidity (line + masked baseline fill)
    m_rh, t_ma, rh_ma = masked_xy(t, rh.values)
    if m_rh.sum() >= 2:
        ax_mid.plot(t[m_rh], rh[m_rh], lw=2.0, color="#6ec6ff", zorder=3)
        zero_ma = np.ma.masked_array(np.zeros_like(rh_ma, dtype=float), mask=rh_ma.mask)
        ax_mid.fill_between(t_ma, rh_ma, zero_ma, where=~t_ma.mask, color=to_rgba("#6ec6ff", 0.22), zorder=1)
        ax_mid.set_ylabel("Relative Humidity (%)")
        ax_mid.set_ylim(0, 100)
    else:
        ax_mid.text(0.02, 0.8, "Humidity: No data available", transform=ax_mid.transAxes, color="#a8b3c3")

    # Bottom: Precipitation + Wind Speed
    m_pr = valid_mask(t, pr.values)
    if m_pr.any():
        ax_bot.bar(
            t[m_pr],
            pr[m_pr],
            width=0.025,
            color=to_rgba("#8be9fd", 0.55),
            edgecolor=to_rgba("#8be9fd", 0.75),
            linewidth=0.4,
            zorder=2,
        )
    else:
        ax_bot.text(0.02, 0.9, "Precipitation: No data available", transform=ax_bot.transAxes, color="#a8b3c3")
    ax_bot.set_ylabel("Precipitation (mm)")

    ax2 = ax_bot.twinx()
    drew_ws = glow_line(ax2, t, ws.values, color="#f8d66d", repeats=10, alpha=0.10, lw=1.8, z=4)
    ax2.set_ylabel("Wind Speed (m/s)")
    if not drew_ws:
        ax2.text(0.02, 0.9, "Wind Speed: No data available", transform=ax2.transAxes, color="#a8b3c3")

    # Axis format
    locator = AutoDateLocator(minticks=4, maxticks=10)
    fmt = DateFormatter("%m-%d\n%H:%M")
    for ax in (ax_top, ax_mid, ax_bot):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(fmt)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Footer
    if len(t) > 0:
        t0, t1 = pd.to_datetime(t.min()), pd.to_datetime(t.max())
        footer = f"Range: {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M}"
    else:
        footer = "Range: No data"
    fig.text(0.015, 0.04, SUBTITLE, ha="left", va="center", fontsize=10, color="#a8b3c3")
    fig.text(0.985, 0.04, footer, ha="right", va="center", fontsize=10, color="#a8b3c3")

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
    fig.savefig(OUT_PATH, dpi=220)
    print(f"Saved: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
