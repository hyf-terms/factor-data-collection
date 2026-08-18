"""Build a common-key pool of historical R-based high-IC factors and strict factors."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


KEYS = ["TRADE_DATE", "SECURITY_ID"]
SOURCES = {
    "strict": (
        "artifacts/effective_factor_correlation/existing_effective_factor_pool.parquet",
        [
            "r43_profitability_persistence_confirmation_equal6",
            "r7_anchor_qop_incremental_w20",
            "r19_multi_horizon_profit",
        ],
    ),
    "quarterly": (
        r"C:\Users\hyf\Desktop\因子\factor_components\quarterly_indicator_candidates.parquet",
        [
            "q1_joint_surprise_growth_cash_60d",
            "q1_all_profit_growth_consistency_60d",
            "q1_margin_cash_improvement_60d",
            "q1_growth_cash_breadth_60d",
            "q1_financial_cash_breadth_60d",
            "q1_cost_discipline_growth_60d",
        ],
    ),
    "raw_q1": (
        r"C:\Users\hyf\Desktop\因子\factor_components\raw_q1_minimal_candidates.parquet",
        ["q1_raw_profit_growth_60d"],
    ),
    "literature": (
        r"C:\Users\hyf\Desktop\因子\factor_components\literature_financial_candidates.parquet",
        ["q1_joint_earnings_revenue", "q1_financial_60d"],
    ),
    "all_quarter": (
        r"C:\Users\hyf\Desktop\因子\factor_components\all_quarter_raw_candidates.parquet",
        ["q1_raw_metric_rd_growth_efficiency_60d"],
    ),
}


def resolved(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    files = []
    for _, (path, columns) in SOURCES.items():
        source = resolved(args.repo, path)
        schema = pq.read_schema(source).names
        missing = [c for c in columns if c not in schema]
        if missing:
            raise ValueError(f"{source} missing {missing}")
        files.append((pq.ParquetFile(source), columns))

    iterators = [f.iter_batches(columns=KEYS + cols, batch_size=750_000) for f, cols in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.parquet")
    writer = None
    try:
        while True:
            batches = []
            for iterator in iterators:
                try:
                    batches.append(next(iterator))
                except StopIteration:
                    batches.append(None)
            if all(b is None for b in batches):
                break
            if any(b is None for b in batches):
                raise RuntimeError("source row counts differ")
            frames = [pa.Table.from_batches([b]).to_pandas() for b in batches]
            base = frames[0][KEYS]
            if any(not base.equals(frame[KEYS]) for frame in frames[1:]):
                raise RuntimeError("source key order differs")
            merged = pd.concat([base] + [frame.drop(columns=KEYS) for frame in frames], axis=1)
            table = pa.Table.from_pandas(merged, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(args.output)
    print(pq.ParquetFile(args.output).metadata.num_rows, pq.read_schema(args.output).names)


if __name__ == "__main__":
    main()
