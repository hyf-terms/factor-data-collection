"""Profitability-state factors from raw level and temporal components.

No prior composite or best factor is used.  Each candidate combines raw
quarterly profitability levels with independently constructed quarterly SUE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from round9_13_ensemble_optimization import KEYS, _robust_z


LEVELS = ["dense_lit_q_gp_assets", "dense_lit_q_op_assets"]
TEMPORAL = ["r15_gp_streak_sue4", "r15_sue_breadth4"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--levels", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel, columns=KEYS)
    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
    levels = pd.read_parquet(args.levels, columns=KEYS + LEVELS)
    temporal = pd.read_parquet(args.temporal, columns=KEYS + TEMPORAL)
    for frame in (levels, temporal):
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
    data = panel.merge(levels, on=KEYS, how="left", validate="one_to_one").merge(
        temporal, on=KEYS, how="left", validate="one_to_one"
    )
    components = LEVELS + TEMPORAL
    data[components] = data[components].fillna(
        data.groupby("TRADE_DATE", sort=False)[components].transform("median")
    ).fillna(0.0)
    z = _robust_z(data, components)
    result = data[KEYS].copy()
    result["r18_gp_level_streak_equal"] = 0.50 * z[LEVELS[0]] + 0.50 * z[TEMPORAL[0]]
    result["r18_gp_level70_streak30"] = 0.70 * z[LEVELS[0]] + 0.30 * z[TEMPORAL[0]]
    result["r18_op_level_breadth_equal"] = 0.50 * z[LEVELS[1]] + 0.50 * z[TEMPORAL[1]]
    result["r18_profitability_state_equal4"] = sum(z[column] for column in components) / 4.0
    factors = [column for column in result if column not in KEYS]
    result[factors] = result[factors].astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(result):,} rows and {len(factors)} standalone state factors")


if __name__ == "__main__":
    main()
