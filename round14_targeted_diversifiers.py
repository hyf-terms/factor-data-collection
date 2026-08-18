"""Test targeted standalone diversifiers for the round 9--13 event factor."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from round9_13_ensemble_optimization import KEYS, _daily_spearman, _orthogonalize, _read_year, _robust_z


EVENT = [
    "r11_core_profit_surprise_assets",
    "r11_revenue_sue",
    "r11_tax_expense_surprise_assets",
]
SECONDARY = [
    "r9_low_share_issuance",
    "r10_beneish_low_sgi",
    "r10_dechow_low_cash_sales_change",
]
NAMES = ["r14c_event_602515"]
for source in ["share", "sgi", "share_sgi", "share_sgi_cashsales"]:
    for weight in [10, 20, 30, 40]:
        NAMES.append(f"r14c_event_{source}_orth_w{weight:02d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "round14_targeted_diversifier_candidates.parquet"
    writer: pq.ParquetWriter | None = None
    training = []
    for year in range(2017, 2027):
        block = pd.read_parquet(args.block_dir / f"year={year}.parquet", columns=KEYS + ["LABEL"])
        r11 = _read_year(
            repo / "artifacts/round11/filled_strict_test/factors_neutralized.parquet",
            EVENT,
            year,
        )
        r9 = _read_year(
            repo / "artifacts/round9/filled_strict_test/factors_neutralized.parquet",
            [SECONDARY[0]],
            year,
        )
        r10 = _read_year(
            repo / "artifacts/round10/sparse_diagnostic_test/factors_neutralized.parquet",
            SECONDARY[1:],
            year,
        )
        data = block.merge(r11, on=KEYS, how="left", validate="one_to_one")
        data = data.merge(r9, on=KEYS, how="left", validate="one_to_one")
        data = data.merge(r10, on=KEYS, how="left", validate="one_to_one")
        z = _robust_z(data, EVENT + SECONDARY)
        data["event"] = z[EVENT].mul([0.60, 0.25, 0.15]).sum(axis=1).astype("float32")
        data["share"] = -z[SECONDARY[0]]
        data["sgi"] = -z[SECONDARY[1]]
        data["cashsales"] = -z[SECONDARY[2]]
        data["share_sgi"] = 0.5 * data["share"] + 0.5 * data["sgi"]
        data["share_sgi_cashsales"] = (
            0.50 * data["share"] + 0.25 * data["sgi"] + 0.25 * data["cashsales"]
        )
        result = data[KEYS].copy()
        result["r14c_event_602515"] = data["event"]
        for source in ["share", "sgi", "share_sgi", "share_sgi_cashsales"]:
            residual = _orthogonalize(data, source, "event")
            for weight in [10, 20, 30, 40]:
                fraction = weight / 100.0
                result[f"r14c_event_{source}_orth_w{weight:02d}"] = (
                    (1.0 - fraction) * data["event"] + fraction * residual
                )
        result[NAMES] = result[NAMES].astype("float32")
        if year <= 2022:
            training.append(
                _daily_spearman(pd.concat([data[KEYS + ["LABEL"]], result[NAMES]], axis=1), NAMES)
            )
        table = pa.Table.from_pandas(result[KEYS + NAMES], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"saved {year}: {len(result):,}")
    if writer is not None:
        writer.close()
    daily = pd.concat(training, ignore_index=True)
    ranking = pd.DataFrame(
        {
            "factor": NAMES,
            "train_mean_ic": [daily[name].mean() for name in NAMES],
            "train_icir": [daily[name].mean() / daily[name].std() for name in NAMES],
        }
    ).sort_values(["train_mean_ic", "train_icir"], ascending=False)
    ranking.to_csv(args.output_dir / "training_candidate_ranking.csv", index=False, encoding="utf-8-sig")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
