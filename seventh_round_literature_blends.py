"""Controlled, label-free blends of the strongest dense literature signals."""

from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dense_q1_gross_profit_factors import robust_daily_zscore
from event_financial_factor_search import KEYS
from third_round_new_factor_optimization import _daily_orthogonal_residual


BASE_DIR = Path(__file__).resolve().parent
ANCHOR = "optimized_interaction"
GP = "dense_lit_q_gp_assets"
OP = "dense_lit_q_op_assets"


def candidate_names() -> list[str]:
    names = [
        "r7_q_profit_equal",
        "r7_q_profit_op60",
        "r7_qop_incremental",
        "r7_qgp_incremental",
        "r7_qprofit_incremental",
    ]
    for source in ("qop", "qgp", "qprofit"):
        names.extend(f"r7_anchor_{source}_incremental_w{weight:02d}" for weight in (5, 10, 15, 20))
    return names


def build_candidates(inputs: pd.DataFrame) -> pd.DataFrame:
    z = robust_daily_zscore(inputs, [ANCHOR, GP, OP])
    anchor = z[ANCHOR]
    gp = z[GP]
    op = z[OP]
    dates = inputs["TRADE_DATE"]
    equal = (gp + op) / 2
    increments = {
        "qop": _daily_orthogonal_residual(op, anchor, dates),
        "qgp": _daily_orthogonal_residual(gp, anchor, dates),
        "qprofit": _daily_orthogonal_residual(equal, anchor, dates),
    }
    result = inputs[KEYS].copy()
    result["r7_q_profit_equal"] = equal
    result["r7_q_profit_op60"] = 0.40 * gp + 0.60 * op
    result["r7_qop_incremental"] = increments["qop"]
    result["r7_qgp_incremental"] = increments["qgp"]
    result["r7_qprofit_incremental"] = increments["qprofit"]
    for source, incremental in increments.items():
        for percent in (5, 10, 15, 20):
            weight = percent / 100
            result[f"r7_anchor_{source}_incremental_w{percent:02d}"] = (
                (1 - weight) * anchor + weight * incremental
            )
    for factor in candidate_names():
        result[factor] = pd.to_numeric(result[factor], errors="coerce").fillna(0.0).astype("float32")
    return result[KEYS + candidate_names()]


def generate(literature_path: Path, anchor_path: Path, output: Path) -> None:
    literature = pd.read_parquet(literature_path, columns=KEYS + [GP, OP])
    anchor = pd.read_parquet(anchor_path, columns=KEYS + [ANCHOR])
    inputs = literature.merge(anchor, on=KEYS, how="inner", validate="one_to_one")
    result = build_candidates(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")


def write_selected(literature_path: Path, blend_path: Path, output: Path) -> None:
    """Stream the retained independent and composite factors into one file."""
    literature_columns = KEYS + [GP, OP]
    blend_columns = KEYS + [
        "r7_anchor_qop_incremental_w20",
        "r7_anchor_qprofit_incremental_w20",
    ]
    left = pq.ParquetFile(literature_path).iter_batches(columns=literature_columns, batch_size=250_000)
    right = pq.ParquetFile(blend_path).iter_batches(columns=blend_columns, batch_size=250_000)
    temporary = output.with_suffix(".tmp.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for left_batch, right_batch in zip_longest(left, right):
            if left_batch is None or right_batch is None or left_batch.num_rows != right_batch.num_rows:
                raise RuntimeError("独立因子与复合因子行数不一致")
            left_table = pa.Table.from_batches([left_batch])
            right_table = pa.Table.from_batches([right_batch])
            left_keys = left_table.select(KEYS).to_pandas()
            right_keys = right_table.select(KEYS).to_pandas()
            if not left_keys.equals(right_keys):
                raise RuntimeError("独立因子与复合因子键顺序不一致")
            table = pa.Table.from_arrays(
                [*left_table.columns, *right_table.select(blend_columns[2:]).columns],
                names=literature_columns + blend_columns[2:],
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有写出第七轮交付因子")
    temporary.replace(output)


def summarize_results(paths: list[Path], base_path: Path, output: Path) -> None:
    frames = [pd.read_parquet(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"])
    base = pd.read_parquet(base_path)
    base = base.loc[base["factor"].eq(ANCHOR), ["TRADE_DATE", "neutral_ic"]].rename(
        columns={"neutral_ic": "base_ic"}
    )
    rows = []
    for factor, group in data.groupby("factor"):
        years = group["TRADE_DATE"].dt.year
        yearly = group.groupby(years)["neutral_ic"].mean()
        joined = group[["TRADE_DATE", "neutral_ic"]].merge(base, on="TRADE_DATE", how="inner")
        rows.append(
            {
                "factor": factor,
                "full_ic": group["neutral_ic"].mean(),
                "train_2017_2022_ic": group.loc[years.between(2017, 2022), "neutral_ic"].mean(),
                "validation_2023_2024_ic": group.loc[years.between(2023, 2024), "neutral_ic"].mean(),
                "historical_check_2025_2026_ic": group.loc[years.between(2025, 2026), "neutral_ic"].mean(),
                "positive_years": int(yearly.gt(0).sum()),
                "minimum_year_ic": yearly.min(),
                "delta_vs_anchor": (joined["neutral_ic"] - joined["base_ic"]).mean(),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("full_ic", ascending=False).to_csv(
        output, index=False, encoding="utf-8-sig"
    )


def main() -> None:
    root = BASE_DIR / "新测试结果" / "第七轮文献稠密财务因子"
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-selected", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--literature", type=Path, default=root / "dense_literature_low_missing_candidates.parquet")
    parser.add_argument("--anchor", type=Path, default=BASE_DIR / "新测试结果" / "第二轮优化" / "round2_selected.parquet")
    parser.add_argument("--output", type=Path, default=root / "dense_literature_blends.parquet")
    parser.add_argument("--selected-output", type=Path, default=root / "round7_selected.parquet")
    parser.add_argument("--independent-daily-ic", type=Path, default=root / "strict_test" / "daily_ic.parquet")
    parser.add_argument("--blend-daily-ic", type=Path, default=root / "blend_strict_test" / "daily_ic.parquet")
    parser.add_argument("--extension-daily-ic", type=Path, default=root / "extension_strict_test" / "daily_ic.parquet")
    parser.add_argument("--base-daily-ic", type=Path, default=BASE_DIR / "新测试结果" / "第二轮优化" / "final_test" / "daily_ic.parquet")
    parser.add_argument("--report", type=Path, default=root / "round7_selection_report.csv")
    args = parser.parse_args()
    if args.write_selected:
        write_selected(args.literature.resolve(), args.output.resolve(), args.selected_output.resolve())
        return
    if args.summarize:
        summarize_results(
            [args.independent_daily_ic.resolve(), args.blend_daily_ic.resolve(), args.extension_daily_ic.resolve()],
            args.base_daily_ic.resolve(),
            args.report.resolve(),
        )
        return
    generate(args.literature.resolve(), args.anchor.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
