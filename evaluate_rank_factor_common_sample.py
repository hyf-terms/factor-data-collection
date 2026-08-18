"""Find IC-qualified low-correlation subsets on an identical date sample."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--daily-ic", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-abs-ic", type=float, default=0.035)
    p.add_argument("--min-common-days", type=int, default=400)
    args = p.parse_args()
    data = pd.read_csv(args.daily_ic)
    factors = [c for c in data if c != "TRADE_DATE" and abs(data[c].mean(skipna=True)) >= args.min_abs_ic]
    rows = []
    for size in range(2, len(factors) + 1):
        for names in combinations(factors, size):
            common = data[list(names)].dropna()
            if len(common) < args.min_common_days:
                continue
            means = common.mean()
            if (means.abs() < args.min_abs_ic).any():
                continue
            corr = common.corr().abs()
            maximum = max(corr.loc[a, b] for a, b in combinations(names, 2))
            rows.append({
                "factor_count": size,
                "factors": "|".join(names),
                "common_days": len(common),
                "min_abs_common_ic": means.abs().min(),
                "max_abs_common_corr": maximum,
            })
    result = pd.DataFrame(rows).sort_values(
        ["factor_count", "max_abs_common_corr", "min_abs_common_ic"],
        ascending=[False, True, False],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    for threshold in [0.85, 0.80, 0.70, 0.60]:
        eligible = result.loc[result["max_abs_common_corr"].lt(threshold)]
        best = eligible.iloc[0].to_dict() if len(eligible) else None
        print(threshold, best)


if __name__ == "__main__":
    main()
