"""
Diagnostic: inspect exactly what schedule_and_record returns, and where
rows are getting lost in build_team_game_log. 31 total games across 30
team-seasons means something is silently dropping ~161 of ~162 games per
team-season -- this print-everything approach finds where.
"""
import pandas as pd
from pybaseball import schedule_and_record

df = schedule_and_record(2023, "NYY")

print("=== Raw shape ===")
print(df.shape)

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== dtypes ===")
print(df.dtypes)

print("\n=== First 10 rows of R, RA, W/L ===")
print(df[["R", "RA", "W/L"]].head(10))

print("\n=== How many R values are actually numeric-looking? ===")
r_numeric = pd.to_numeric(df["R"], errors="coerce")
print(f"Non-null after to_numeric: {r_numeric.notna().sum()} out of {len(df)}")

print("\n=== Sample of raw R values (unique, first 20) ===")
print(df["R"].unique()[:20])
