"""Refine a standalone round 9--13 factor without any legacy anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from round9_13_ensemble_optimization import KEYS, _daily_spearman, _orthogonalize, _read_year, _robust_z


EVENT_COLUMNS = [
    "r11_core_profit_surprise_assets",
    "r11_revenue_sue",
    "r11_tax_expense_surprise_assets",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def candidate_names() -> list[str]:
    names = []
    for event in ["ic", "503020", "602515"]:
        names.append(f"r14b_event_{event}")
        for weight in [20, 25, 30, 35, 40, 50]:
            names.append(f"r14b_event_{event}_finorth_w{weight:02d}")
    names.extend(
        [
            "r14b_event_ic_fin20_mis10",
            "r14b_event_ic_fin25_mis10",
            "r14b_event_ic_fin30_mis10",
            "r14b_event_ic_fin25_mis15",
            "r14b_event_ic_fin25_mis10_emp05",
        ]
    )
    return names


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    direction = pd.read_csv(repo / "artifacts/round14/component_training_directions.csv")
    event_stats = direction.set_index("factor").loc[EVENT_COLUMNS]
    event_weights = event_stats["train_mean_ic_raw"].clip(lower=0.0)
    event_weights = event_weights / event_weights.sum()
    fixed_weights = {
        "ic": event_weights.to_list(),
        "503020": [0.50, 0.30, 0.20],
        "602515": [0.60, 0.25, 0.15],
    }

    names = candidate_names()
    training_ic_parts = []
    output = args.output_dir / "round14_independent_refined_candidates.parquet"
    writer: pq.ParquetWriter | None = None
    for year in range(2017, 2027):
        block = pd.read_parquet(args.block_dir / f"year={year}.parquet")
        event_raw = _read_year(
            repo / "artifacts/round11/filled_strict_test/factors_neutralized.parquet",
            EVENT_COLUMNS,
            year,
        )
        data = block.merge(event_raw, on=KEYS, how="left", validate="one_to_one")
        event_z = _robust_z(data, EVENT_COLUMNS)
        result = data[KEYS].copy()
        for event_name, weights in fixed_weights.items():
            event_column = f"event_{event_name}"
            data[event_column] = event_z.mul(weights).sum(axis=1).astype("float32")
            fin_orth = _orthogonalize(data, "financing", event_column)
            result[f"r14b_event_{event_name}"] = data[event_column]
            for weight in [20, 25, 30, 35, 40, 50]:
                fraction = weight / 100.0
                result[f"r14b_event_{event_name}_finorth_w{weight:02d}"] = (
                    (1.0 - fraction) * data[event_column] + fraction * fin_orth
                )

        anchor = data["event_ic"]
        fin = _orthogonalize(data, "financing", "event_ic")
        mis = _orthogonalize(data, "misstatement", "event_ic")
        emp = _orthogonalize(data, "employee", "event_ic")
        result["r14b_event_ic_fin20_mis10"] = 0.70 * anchor + 0.20 * fin + 0.10 * mis
        result["r14b_event_ic_fin25_mis10"] = 0.65 * anchor + 0.25 * fin + 0.10 * mis
        result["r14b_event_ic_fin30_mis10"] = 0.60 * anchor + 0.30 * fin + 0.10 * mis
        result["r14b_event_ic_fin25_mis15"] = 0.60 * anchor + 0.25 * fin + 0.15 * mis
        result["r14b_event_ic_fin25_mis10_emp05"] = (
            0.60 * anchor + 0.25 * fin + 0.10 * mis + 0.05 * emp
        )
        result[names] = result[names].astype("float32")
        if year <= 2022:
            train_frame = pd.concat([data[KEYS + ["LABEL"]], result[names]], axis=1)
            training_ic_parts.append(_daily_spearman(train_frame, names))
        table = pa.Table.from_pandas(result[KEYS + names], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"saved {year}: {len(result):,}")
    if writer is not None:
        writer.close()

    daily = pd.concat(training_ic_parts, ignore_index=True)
    summary = pd.DataFrame(
        {
            "factor": names,
            "train_mean_ic": [daily[name].mean() for name in names],
            "train_icir": [daily[name].mean() / daily[name].std() for name in names],
        }
    ).sort_values(["train_mean_ic", "train_icir"], ascending=False)
    summary.to_csv(
        args.output_dir / "training_candidate_ranking.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "uses_legacy_anchor": False,
                "training_period": "2017-2022",
                "event_ic_weights": dict(zip(EVENT_COLUMNS, event_weights.to_list())),
                "candidate_columns": names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.head(12).to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
