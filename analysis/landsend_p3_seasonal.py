"""Seasonal analysis of S2 band signals at Landsend presence 3.

Fetches all cloud-acceptable S2 L2A observations for the presence-3 patch
over 2021-2025, computes pixel-median per date, then plots a smoothed
timeseries (30-day rolling median) for the bands most likely to reflect
a yellow-shift: B04, B05, B06, B07, NDVI, red-edge NDVI, CIre.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pixel_collector import collect
from analysis.constants import SCL_CLEAR_VALUES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Config ──────────────────────────────────────────────────────────────────
BBOX = [141.391728, -20.456644, 141.398662, -20.454502]
START = "2021-01-01"
END   = "2025-12-31"
CLOUD_MAX = 30
OUT_DIR = PROJECT_ROOT / "data" / "pixels" / "landsend_p3_seasonal"
CACHE_DIR = PROJECT_ROOT / "data" / "chips" / "landsend_p3_seasonal"
PLOT_OUT = PROJECT_ROOT / "outputs" / "landsend_p3_seasonal.png"

PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Fetch / load ─────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
parquet_files = list(OUT_DIR.glob("*.parquet"))

if not parquet_files:
    logging.info("No cached data — fetching via pixel_collector …")
    parquet_files = collect(
        bbox_wgs84=BBOX,
        start=START,
        end=END,
        out_dir=OUT_DIR,
        cloud_max=CLOUD_MAX,
        cache_dir=CACHE_DIR,
        apply_nbar=True,
    )
    # remove the coords parquet from analysis list
    parquet_files = [p for p in parquet_files if "coords" not in p.name]
else:
    logging.info("Using cached pixel data (%d tile files)", len(parquet_files))
    parquet_files = [p for p in parquet_files if "coords" not in p.name]

if not parquet_files:
    logging.error("No parquet files found after collection — exiting")
    sys.exit(1)

df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
logging.info("Raw rows: %d", len(df))

# ── Quality filter ────────────────────────────────────────────────────────────
# Keep only clear pixels (SCL 4/5/6/7/11) and decent scl_purity
clear_scl = list(SCL_CLEAR_VALUES)
df = df[df["scl"].isin(clear_scl) & (df["scl_purity"] >= 0.5)]
logging.info("After quality filter: %d rows", len(df))

df["date"] = pd.to_datetime(df["date"])

# ── Derived indices ───────────────────────────────────────────────────────────
# red-edge NDVI: (B8A - B05) / (B8A + B05)
df["re_ndvi"] = (df["B8A"] - df["B05"]) / (df["B8A"] + df["B05"] + 1e-6)
# CIre: B07/B05 - 1
df["cire"] = df["B07"] / (df["B05"] + 1e-6) - 1.0

# ── Per-date pixel median ─────────────────────────────────────────────────────
bands_of_interest = ["B04", "B05", "B06", "B07", "B8A", "NDVI", "re_ndvi", "cire"]
daily = (
    df.groupby("date")[bands_of_interest]
    .median()
    .reset_index()
    .sort_values("date")
)
logging.info("Unique dates: %d", len(daily))

# ── Smooth: Gaussian-weighted rolling mean on the actual date axis ────────────
# ~4 obs/month means a hard rolling window leaves many gaps; instead resample
# to daily (forward-fill to at most 5 days) then apply a 45-day Gaussian kernel.
daily = daily.set_index("date").sort_index()
daily_resampled = daily[bands_of_interest].resample("D").median()
# Gaussian kernel: std=15 days → FWHM ~35 days; min_periods=2 so short gaps fill
smoothed = daily_resampled.rolling(90, center=True, min_periods=2, win_type="gaussian").mean(std=15)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
fig.suptitle("Landsend presence 3 — S2 timeseries 2021–2025 (30-day rolling median)", fontsize=13)

plot_items = [
    ("B04", "red",         "B04 red reflectance"),
    ("B05", "tomato",      "B05 red-edge 1 reflectance"),
    ("B06", "darkorange",  "B06 red-edge 2 reflectance"),
    ("B07", "saddlebrown", "B07 red-edge 3 reflectance"),
    ("B8A", "darkgreen",   "B8A NIR narrow reflectance"),
    ("NDVI","green",       "NDVI"),
    ("re_ndvi","olive",    "Red-edge NDVI (B8A−B05)/(B8A+B05)"),
    ("cire","goldenrod",   "CIre  B07/B05 − 1"),
]

# shade each dry season window (May–Oct) across all years
def _shade_dry(ax, start_year, end_year):
    for yr in range(start_year, end_year + 1):
        ax.axvspan(
            pd.Timestamp(f"{yr}-05-01"), pd.Timestamp(f"{yr}-10-31"),
            color="khaki", alpha=0.30, linewidth=0,
        )

for ax, (col, colour, title) in zip(axes.flat, plot_items):
    _shade_dry(ax, 2021, 2025)
    ax.plot(smoothed.index, smoothed[col], color=colour, linewidth=1.8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

for ax in axes[-1]:
    ax.set_xlabel("")

handles = [plt.Rectangle((0, 0), 1, 1, color="khaki", alpha=0.5)]
fig.legend(handles, ["dry season (May–Oct)"], loc="lower center", ncol=1, fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(PLOT_OUT, dpi=150)
logging.info("Saved plot → %s", PLOT_OUT)
