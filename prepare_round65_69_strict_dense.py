"""Prepare strict full-panel candidates after sparse testing.

Missing companies are filled with the same-date cross-sectional median.  A
factor is rejected in full (never date-skipped) if any date has no observation
or if any filled daily cross-section is constant.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["TRADE_DATE", "SECURITY_ID"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sparse", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-missing-rate", type=float, default=0.10)
    a = p.parse_args()
    data = pd.read_parquet(a.sparse)
    factors = [c for c in data.columns if c not in KEYS]
    accepted, report = [], []
    for factor in factors:
        values = pd.to_numeric(data[factor], errors="coerce")
        by_date = values.groupby(data.TRADE_DATE, sort=False)
        observed = by_date.count()
        whole_empty = int(observed.eq(0).sum())
        medians = by_date.transform("median")
        filled = values.fillna(medians)
        constant = int(filled.groupby(data.TRADE_DATE, sort=False).nunique().lt(2).sum())
        missing_rate = float(values.isna().mean())
        status = "accepted" if (
            whole_empty == 0 and constant == 0 and missing_rate <= a.max_missing_rate
        ) else "rejected"
        report.append({
            "factor": factor,
            "status": status,
            "missing_rate_before_fill": missing_rate,
            "whole_empty_days": whole_empty,
            "constant_days_after_fill": constant,
        })
        if status == "accepted":
            data[factor] = filled.astype("float32")
            accepted.append(factor)
    if not accepted:
        raise RuntimeError("no factor satisfies strict full-date eligibility")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    data[KEYS + accepted].to_parquet(a.output, index=False, compression="zstd")
    pd.DataFrame(report).to_csv(a.report, index=False, encoding="utf-8-sig")
    print(f"accepted={len(accepted)}, rejected={len(factors)-len(accepted)}")


if __name__ == "__main__":
    main()
