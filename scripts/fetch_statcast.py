"""Download one MLB regular season of pitch-level Statcast data from Baseball Savant.

Usage: python scripts/fetch_statcast.py 2025
Writes data/raw/statcast_<year>.parquet (only the columns the model needs).
"""
import sys
from pathlib import Path

from pybaseball import cache, statcast

SEASONS = {2024: ("2024-03-20", "2024-09-30"), 2025: ("2025-03-18", "2025-09-28")}
KEEP = ["game_date", "pitcher", "player_name", "p_throws", "pitch_type",
        "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
        "spin_axis", "release_pos_x", "release_pos_z", "release_extension"]

year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
start, end = SEASONS[year]
cache.enable()
df = statcast(start_dt=start, end_dt=end)[KEEP]
df = df[df["game_date"] >= start]  # drop any spring games pybaseball returns
out = Path(__file__).resolve().parents[1] / "data" / "raw" / f"statcast_{year}.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print(f"{len(df):,} pitches from {df['pitcher'].nunique():,} pitchers -> {out}")
