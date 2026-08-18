"""Controlled incremental blends for the strongest round-eight signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dense_q1_gross_profit_factors import robust_daily_zscore
from event_financial_factor_search import KEYS
from third_round_new_factor_optimization import _daily_orthogonal_residual


BASE_DIR = Path(__file__).resolve().parent
ANCHOR = "optimized_interaction"
SGA = "r8_ab_sales_sga_gap"
AR = "r8_ab_sales_receivable_gap"


def candidate_names() -> list[str]:
    names = ["r8_ab_sga_incremental", "r8_ab_ar_incremental", "r8_ab_gap_incremental"]
    for source in ("sga", "ar", "gap"):
        names.extend(f"r8_anchor_ab_{source}_w{weight:02d}" for weight in (5, 10, 15, 20))
    return names


def build_candidates(inputs: pd.DataFrame) -> pd.DataFrame:
    z = robust_daily_zscore(inputs, [ANCHOR, SGA, AR])
    anchor = z[ANCHOR]
    dates = inputs["TRADE_DATE"]
    sga_incremental = _daily_orthogonal_residual(z[SGA], anchor, dates)
    ar_incremental = _daily_orthogonal_residual(z[AR], anchor, dates)
    gap_incremental = _daily_orthogonal_residual((z[SGA] + z[AR]) / 2, anchor, dates)
    increments = {
        "sga": sga_incremental,
        "ar": ar_incremental,
        "gap": gap_incremental,
    }
    result = inputs[KEYS].copy()
    result["r8_ab_sga_incremental"] = sga_incremental
    result["r8_ab_ar_incremental"] = ar_incremental
    result["r8_ab_gap_incremental"] = gap_incremental
    for source, incremental in increments.items():
        for percent in (5, 10, 15, 20):
            weight = percent / 100
            result[f"r8_anchor_ab_{source}_w{percent:02d}"] = (
                (1 - weight) * anchor + weight * incremental
            )
    for factor in candidate_names():
        result[factor] = pd.to_numeric(result[factor], errors="coerce").fillna(0.0).astype("float32")
    return result[KEYS + candidate_names()]


def generate(round8_path: Path, anchor_path: Path, output: Path) -> None:
    factors = pd.read_parquet(round8_path, columns=KEYS + [SGA, AR])
    anchor = pd.read_parquet(anchor_path, columns=KEYS + [ANCHOR])
    inputs = factors.merge(anchor, on=KEYS, how="inner", validate="one_to_one")
    result = build_candidates(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")


def parse_args() -> argparse.Namespace:
    root = BASE_DIR / "新测试结果" / "第八轮新增文献财务因子"
    parser = argparse.ArgumentParser()
    parser.add_argument("--round8", type=Path, default=root / "round8_filled_after_test.parquet")
    parser.add_argument("--anchor", type=Path, default=BASE_DIR / "新测试结果" / "第二轮优化" / "round2_selected.parquet")
    parser.add_argument("--output", type=Path, default=root / "round8_incremental_blends.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.round8.resolve(), args.anchor.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
