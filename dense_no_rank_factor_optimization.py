"""Dense optimizations for retained no-rank financial signals.

The grid is deliberately small: Q1 weights 0.25/0.50 and either no decay or
a 60-trading-day half-life.  Each target family is shrunk 25% toward the
high-coverage gross-profit-growth signal.  Construction is label-free, source
missingness is mapped to neutral zero only after daily robust standardization,
and every final candidate must vary on every test date.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from all_quarter_raw_factor_search import (
    SIGNAL_COLUMNS,
    attach_prior_year,
    calculate_event_signals,
    load_all_quarter_events,
)
from dense_q1_gross_profit_factors import robust_daily_zscore, select_parameters
from event_financial_factor_search import (
    KEYS,
    _map_events,
    _normalize_panel,
    _prepare_events,
    build_deducted_sue_events,
)
from literature_financial_factor_search import build_income_surprise_events
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS
from quarterly_indicator_factor_search import map_events_to_panel


BASE_DIR = Path(__file__).resolve().parent
Q1_WEIGHTS = (0.25, 0.50)
HALF_LIVES: tuple[int | None, ...] = (None, 60)
ANCHOR_SIGNAL = "GROSS_PROFIT_GROWTH"
ANCHOR_WEIGHT = 0.25

RAW_SIGNALS = [
    "PROFIT_GROWTH",
    "GROWTH_CASH_BREADTH",
    "MARGIN_CASH_IMPROVEMENT",
    "RD_GROWTH_EFFICIENCY",
    "GROSS_PROFIT_GROWTH",
]
SUE_SIGNALS = [
    "NET_INCOME_SUE",
    "DEDUCTED_INCOME_SUE",
    "OPERATING_PROFIT_SUE",
    "GROSS_PROFIT_SUE",
]
BASE_SIGNALS = RAW_SIGNALS + SUE_SIGNALS

FAMILIES = {
    "profit_growth": ["PROFIT_GROWTH"],
    "growth_cash_breadth": ["GROWTH_CASH_BREADTH"],
    "margin_cash_improvement": ["MARGIN_CASH_IMPROVEMENT"],
    "rd_growth_efficiency": ["RD_GROWTH_EFFICIENCY"],
    "net_income_sue": ["NET_INCOME_SUE"],
    "deducted_income_sue": ["DEDUCTED_INCOME_SUE"],
    "operating_profit_sue": ["OPERATING_PROFIT_SUE"],
    "gross_profit_sue": ["GROSS_PROFIT_SUE"],
    "earnings_sue_ensemble": SUE_SIGNALS,
    "quality_growth_ensemble": [
        "GROSS_PROFIT_GROWTH",
        "PROFIT_GROWTH",
        "GROWTH_CASH_BREADTH",
        "MARGIN_CASH_IMPROVEMENT",
        "NET_INCOME_SUE",
        "OPERATING_PROFIT_SUE",
    ],
}


def _mean_complete(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].mean(axis=1, skipna=False)


def prepare_raw_events(
    data_root: Path, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    data = calculate_event_signals(
        attach_prior_year(load_all_quarter_events(data_root))
    )
    growth_fields = [
        "REVENUE_GROWTH",
        "PROFIT_GROWTH",
        "CFO_GROWTH",
        "SALES_CASH_GROWTH",
    ]
    data["GROWTH_CASH_BREADTH"] = _mean_complete(data, growth_fields)
    data["MARGIN_CASH_IMPROVEMENT"] = _mean_complete(
        data, ["GROSS_MARGIN_CHANGE", "PROFIT_GROWTH", "CFO_GROWTH"]
    )
    data["RD_GROWTH_EFFICIENCY"] = _mean_complete(
        data, ["RD_GROWTH", "REVENUE_GROWTH", "PROFIT_GROWTH"]
    )
    available = assign_available_trade_date(data, calendar)
    available = available.sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
    )
    newest = available.groupby("SECURITY_ID", sort=False)[
        "QUARTER_INDEX"
    ].cummax()
    clean = available.loc[available["QUARTER_INDEX"].eq(newest)].copy()
    clean = clean.drop_duplicates(
        ["SECURITY_ID", "AVAILABLE_DATE"], keep="last"
    )
    return clean[
        [
            "SECURITY_ID",
            "AVAILABLE_DATE",
            "FISCAL_QUARTER",
            "QUARTER_INDEX",
            *RAW_SIGNALS,
        ]
    ].reset_index(drop=True)


def prepare_sue_groups(
    base_dir: Path, calendar: pd.DatetimeIndex
) -> dict[str, dict[int, pd.DataFrame]]:
    pead = pd.read_parquet(
        base_dir / "factor_components" / "pead_sue_events.parquet"
    )
    indicator = pd.read_parquet(base_dir / "data" / "ch_models" / "earnings_pit")
    deducted = build_deducted_sue_events(indicator)
    income = pd.read_parquet(
        base_dir / "data" / "new_pit" / "new_pit_income",
        columns=COMMON_COLUMNS + ["REVENUE", "OPERATE_PROFIT", "COGS"],
    )
    surprises = build_income_surprise_events(income)
    specs = {
        "NET_INCOME_SUE": (pead, "SUE_RAW"),
        "DEDUCTED_INCOME_SUE": (deducted, "SUE_RAW"),
        "OPERATING_PROFIT_SUE": (surprises["operate"], "SUE_RAW"),
        "GROSS_PROFIT_SUE": (surprises["gross"], "SUE_RAW"),
    }
    return {
        name: _prepare_events(events, calendar, value_column=value)
        for name, (events, value) in specs.items()
    }


def grid_specs() -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for weight in Q1_WEIGHTS:
            for half_life in HALF_LIVES:
                weight_code = int(round(weight * 100))
                decay_code = "none" if half_life is None else f"h{half_life}"
                rows.append(
                    {
                        "factor": (
                            f"grid_dense_{family}_{decay_code}_q1w{weight_code:03d}"
                        ),
                        "version": family,
                        "q1_weight": weight,
                        "half_life": half_life,
                    }
                )
    return pd.DataFrame(rows)


def map_signal_inputs(
    panel: pd.DataFrame,
    raw_events: pd.DataFrame,
    sue_groups: dict[str, dict[int, pd.DataFrame]],
    calendar_ordinal: pd.Series,
) -> pd.DataFrame:
    mapped = map_events_to_panel(panel, raw_events, calendar_ordinal)
    mapped = mapped.rename(
        columns={
            "FISCAL_QUARTER": "RAW_QUARTER",
            "EVENT_AGE": "RAW_AGE",
        }
    )
    keep = KEYS + ["RAW_QUARTER", "RAW_AGE", *RAW_SIGNALS]
    result = mapped[keep].copy()
    for signal in SUE_SIGNALS:
        prefix = signal.lower()
        one = _map_events(
            panel,
            sue_groups[signal],
            calendar_ordinal,
            "SUE_RAW",
            prefix,
        ).rename(
            columns={
                f"{prefix}_value": signal,
                f"{prefix}_quarter": f"{signal}_QUARTER",
                f"{prefix}_age": f"{signal}_AGE",
            }
        )
        result = result.merge(one, on=KEYS, how="left", validate="one_to_one")
    return result


def _quarter_and_age(data: pd.DataFrame, signal: str) -> tuple[pd.Series, pd.Series]:
    if signal in RAW_SIGNALS:
        return data["RAW_QUARTER"], data["RAW_AGE"]
    return data[f"{signal}_QUARTER"], data[f"{signal}_AGE"]


def build_grid_for_slice(data: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    z = robust_daily_zscore(data, BASE_SIGNALS)
    result = data[KEYS].copy()

    def transform(signal: str, q1_weight: float, half_life: float) -> pd.Series:
        quarter, age = _quarter_and_age(data, signal)
        value = z[signal]
        if pd.notna(half_life):
            decay = np.exp(
                -np.log(2.0)
                * pd.to_numeric(age, errors="coerce").clip(lower=0)
                / float(half_life)
            ).fillna(0.0)
            value = value * decay
        return value * (
            1.0 + float(q1_weight) * quarter.eq(1).astype("float64")
        )

    for row in specs.itertuples(index=False):
        components = [
            transform(signal, row.q1_weight, row.half_life)
            for signal in FAMILIES[row.version]
        ]
        target = pd.concat(components, axis=1).mean(axis=1, skipna=False)
        anchor = transform(ANCHOR_SIGNAL, row.q1_weight, row.half_life)
        combined = (1.0 - ANCHOR_WEIGHT) * target + ANCHOR_WEIGHT * anchor
        result[row.factor] = combined.fillna(0.0).astype("float32")
    return result


def _quality(candidates: pd.DataFrame, specs: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    for factor in specs["factor"]:
        grouped = candidates.groupby("TRADE_DATE", sort=False)[factor]
        spread = grouped.max() - grouped.min()
        rows.append(
            {
                "year": year,
                "factor": factor,
                "nonnull_ratio": float(candidates[factor].notna().mean()),
                "constant_days": int(spread.le(1e-10).sum()),
                "calendar_days": int(candidates.TRADE_DATE.nunique()),
            }
        )
    return pd.DataFrame(rows)


def generate_grid(
    panel_path: Path,
    data_root: Path,
    output_path: Path,
    manifest_path: Path,
    quality_path: Path,
) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique())
    calendar = calendar.normalize().drop_duplicates().sort_values()
    ordinal = pd.Series(np.arange(len(calendar), dtype=np.int16), index=calendar)
    raw_events = prepare_raw_events(data_root, calendar)
    sue_groups = prepare_sue_groups(BASE_DIR, calendar)
    specs = grid_specs()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    specs.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    quality_frames = []
    try:
        for year in sorted(set(calendar.year)):
            panel = pd.read_parquet(
                panel_path,
                columns=KEYS,
                filters=[
                    ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
                    ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
                ],
            )
            panel = _normalize_panel(panel)
            inputs = map_signal_inputs(panel, raw_events, sue_groups, ordinal)
            candidates = build_grid_for_slice(inputs, specs)
            quality = _quality(candidates, specs, year)
            quality_frames.append(quality)
            bad = quality.loc[quality.constant_days.gt(0)]
            if not bad.empty:
                raise RuntimeError(
                    f"{year}存在恒定日，拒绝输出: "
                    f"{bad[['factor','constant_days']].head().to_dict('records')}"
                )
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd"
                )
            writer.write_table(table)
            print(f"{year}: rows={len(candidates):,}, factors={len(specs)}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成候选")
    os.replace(temporary, output_path)
    pd.concat(quality_frames, ignore_index=True).to_csv(
        quality_path, index=False, encoding="utf-8-sig"
    )


def write_selected(
    grid_path: Path, selected: pd.DataFrame, output_path: Path
) -> None:
    source = selected["factor"].tolist()
    names = [f"dense_q1_{version}" for version in selected["version"]]
    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    parquet = pq.ParquetFile(grid_path)
    try:
        for batch in parquet.iter_batches(columns=KEYS + source, batch_size=250_000):
            table = pa.Table.from_batches([batch]).rename_columns(KEYS + names)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("参数网格为空")
    os.replace(temporary, output_path)


def fit(
    daily_ic_path: Path,
    manifest_path: Path,
    grid_path: Path,
    output_path: Path,
    report_path: Path,
    parameters_path: Path,
) -> None:
    daily = pd.read_parquet(daily_ic_path)
    specs = pd.read_csv(manifest_path)
    selected, report = select_parameters(daily, specs)
    write_selected(grid_path, selected, output_path)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    parameters_path.write_text(
        json.dumps(
            selected[
                ["version", "factor", "q1_weight", "half_life", "cv_score"]
            ].to_dict("records"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-grid")
    generate.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    generate.add_argument(
        "--data-root", type=Path, default=BASE_DIR / "data" / "new_pit"
    )
    generate.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_grid.parquet",
    )
    generate.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_grid.csv",
    )
    generate.add_argument(
        "--quality",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_grid_quality.csv",
    )
    fitted = sub.add_parser("fit")
    fitted.add_argument("--daily-ic", type=Path, required=True)
    fitted.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_grid.csv",
    )
    fitted.add_argument(
        "--grid",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_grid.parquet",
    )
    fitted.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_factors.parquet",
    )
    fitted.add_argument(
        "--report",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_parameter_report.csv",
    )
    fitted.add_argument(
        "--parameters",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_parameters.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-grid":
        generate_grid(
            args.panel.resolve(),
            args.data_root.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
            args.quality.resolve(),
        )
    else:
        fit(
            args.daily_ic.resolve(),
            args.manifest.resolve(),
            args.grid.resolve(),
            args.output.resolve(),
            args.report.resolve(),
            args.parameters.resolve(),
        )


if __name__ == "__main__":
    main()
