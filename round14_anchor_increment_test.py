"""Add round 9--13 ensemble residuals to the existing round-7 anchor.

The candidate ensemble is orthogonalized cross-sectionally to the anchor on
each date.  Only the 2017--2022 label determines the residual direction.  The
tested weights are fixed ex ante at 5%, 10%, and 20%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from round9_13_ensemble_optimization import (
    KEYS,
    _daily_spearman,
    _orthogonalize,
    _read_year,
    _robust_z,
)


ANCHOR = "r7_anchor_qop_incremental_w20"
ENSEMBLES = [
    "r14_event_finorth_w20",
    "r14_event_multiorth",
    "r14_train_ic_weighted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_year(args: argparse.Namespace, year: int) -> pd.DataFrame:
    anchor = _read_year(args.anchor, [ANCHOR], year)
    ensemble = _read_year(args.ensemble, ENSEMBLES, year)
    label = _read_year(args.label, ["LABEL"], year)
    result = ensemble.merge(anchor, on=KEYS, how="left", validate="one_to_one")
    result = result.sort_values(["SECURITY_ID", "TRADE_DATE"])
    result[ANCHOR] = result.groupby("SECURITY_ID", sort=False)[ANCHOR].ffill()
    return result.merge(label, on=KEYS, how="left", validate="one_to_one")


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    z = _robust_z(frame, [ANCHOR] + ENSEMBLES)
    result = pd.concat([frame[KEYS + ["LABEL"]], z], axis=1)
    for ensemble in ENSEMBLES:
        result[f"{ensemble}_orth"] = _orthogonalize(result, ensemble, ANCHOR)
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[int, pd.DataFrame] = {}
    train_ic = []
    residual_columns = [f"{name}_orth" for name in ENSEMBLES]
    for year in range(2017, 2027):
        frame = prepare(load_year(args, year))
        prepared[year] = frame
        if year <= 2022:
            train_ic.append(_daily_spearman(frame, residual_columns))
    daily = pd.concat(train_ic, ignore_index=True)
    directions = {
        column: (1.0 if daily[column].mean() >= 0 else -1.0)
        for column in residual_columns
    }
    pd.DataFrame(
        {
            "residual": residual_columns,
            "train_mean_ic": [daily[c].mean() for c in residual_columns],
            "direction": [directions[c] for c in residual_columns],
        }
    ).to_csv(
        args.output_dir / "anchor_increment_training_directions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output = args.output_dir / "round14_anchor_increment_candidates.parquet"
    writer: pq.ParquetWriter | None = None
    factor_columns = ["r14_anchor_baseline"]
    for ensemble in ENSEMBLES:
        short = ensemble.removeprefix("r14_")
        factor_columns.append(f"r14_anchor_{short}_direct20")
        factor_columns.extend(
            [f"r14_anchor_{short}_orth_w{weight:02d}" for weight in [5, 10, 20]]
        )

    for year in range(2017, 2027):
        frame = prepared.pop(year)
        result = frame[KEYS].copy()
        anchor = frame[ANCHOR]
        result["r14_anchor_baseline"] = anchor
        for ensemble in ENSEMBLES:
            short = ensemble.removeprefix("r14_")
            result[f"r14_anchor_{short}_direct20"] = 0.80 * anchor + 0.20 * frame[ensemble]
            residual = directions[f"{ensemble}_orth"] * frame[f"{ensemble}_orth"]
            for weight in [5, 10, 20]:
                fraction = weight / 100.0
                result[f"r14_anchor_{short}_orth_w{weight:02d}"] = (
                    (1.0 - fraction) * anchor + fraction * residual
                )
        result[factor_columns] = result[factor_columns].astype("float32")
        table = pa.Table.from_pandas(result[KEYS + factor_columns], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"saved year {year}: {len(result):,}")
    if writer is not None:
        writer.close()
    (args.output_dir / "anchor_increment_metadata.json").write_text(
        json.dumps(
            {
                "anchor": ANCHOR,
                "ensembles": ENSEMBLES,
                "directions": directions,
                "factor_columns": factor_columns,
                "training_period": "2017-2022",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
