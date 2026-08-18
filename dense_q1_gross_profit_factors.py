"""Build and fit dense Q1-enhanced gross-profit factors without look-ahead.

Candidate construction never reads labels.  Every candidate is defined on the
complete factor panel: the latest disclosed Q1-Q4 value is carried forward,
cross-sectionally standardized, and source-missing observations are assigned
the neutral standardized value zero.  Q1 is an economically pre-specified
interaction, not a calendar-date filter.

Parameter fitting reads only daily IC produced by ``factors_neus_only.py`` and
uses 2017-2022 annual folds.  Validation and holdout dates are reported only
after parameters have been frozen.
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

from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date, build_sue_events
from quarterly_f_score import COMMON_COLUMNS, build_standalone_quarterly_metric
from quarterly_indicator_factor_search import map_events_to_panel


BASE_DIR = Path(__file__).resolve().parent
Q1_WEIGHTS = (0.25, 0.50, 1.00)
HALF_LIVES = (20, 40, 60)
TRAIN_END = pd.Timestamp("2022-12-31")
VALIDATION_START = pd.Timestamp("2023-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")

FINAL_NAMES = {
    "gp_level_no_decay": "dense_q1_gp_level",
    "gp_level_decay": "dense_q1_gp_level_decay",
    "gp_acceleration": "dense_q1_gp_acceleration",
    "gp_sue_decay": "dense_q1_gp_sue_decay",
}


def _load_income(data_root: Path) -> pd.DataFrame:
    columns = [*COMMON_COLUMNS, "REVENUE", "COGS"]
    return pd.read_parquet(data_root / "new_pit_income", columns=columns)


def build_gross_profit_events(data_root: Path) -> pd.DataFrame:
    """Build PIT-safe level-growth, acceleration and SUE event values."""
    income = _load_income(data_root)
    revenue = build_standalone_quarterly_metric(
        income, "REVENUE", name="income:REVENUE"
    )
    cogs = build_standalone_quarterly_metric(
        income, "COGS", name="income:COGS"
    )
    keys = ["SECURITY_ID", "FISCAL_YEAR", "FISCAL_QUARTER", "QUARTER_INDEX"]
    merged = revenue.merge(
        cogs,
        on=keys,
        how="inner",
        suffixes=("_REVENUE", "_COGS"),
        validate="one_to_one",
    )
    quarterly = merged[keys].copy()
    quarterly["END_DATE"] = pd.concat(
        [
            pd.to_datetime(merged["END_DATE_REVENUE"], errors="coerce"),
            pd.to_datetime(merged["END_DATE_COGS"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    quarterly["EVENT_TIME"] = pd.concat(
        [
            pd.to_datetime(merged["EVENT_TIME_REVENUE"], errors="coerce"),
            pd.to_datetime(merged["EVENT_TIME_COGS"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    quarterly["GROSS_PROFIT"] = (
        pd.to_numeric(merged["REVENUE"], errors="coerce")
        - pd.to_numeric(merged["COGS"], errors="coerce")
    )

    lag4 = quarterly[
        ["SECURITY_ID", "QUARTER_INDEX", "GROSS_PROFIT", "EVENT_TIME"]
    ].copy()
    lag4["QUARTER_INDEX"] += 4
    lag4 = lag4.rename(
        columns={
            "GROSS_PROFIT": "GROSS_PROFIT_LAG4",
            "EVENT_TIME": "LAG4_EVENT_TIME",
        }
    )
    events = quarterly.merge(
        lag4,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="left",
        validate="one_to_one",
    )
    denominator = events["GROSS_PROFIT_LAG4"].abs().where(
        events["GROSS_PROFIT_LAG4"].abs().gt(1e-12)
    )
    events["GP_GROWTH"] = (
        events["GROSS_PROFIT"] - events["GROSS_PROFIT_LAG4"]
    ).div(denominator)
    events["EVENT_TIME"] = pd.concat(
        [
            pd.to_datetime(events["EVENT_TIME"], errors="coerce"),
            pd.to_datetime(events["LAG4_EVENT_TIME"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)

    prior = events[["SECURITY_ID", "QUARTER_INDEX", "GP_GROWTH"]].copy()
    prior["QUARTER_INDEX"] += 1
    prior = prior.rename(columns={"GP_GROWTH": "GP_GROWTH_PREV_Q"})
    events = events.merge(
        prior,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="left",
        validate="one_to_one",
    )
    events["GP_ACCELERATION"] = (
        events["GP_GROWTH"] - events["GP_GROWTH_PREV_Q"]
    )

    sue_input = quarterly.rename(
        columns={"GROSS_PROFIT": "QUARTERLY_EARNINGS"}
    ).copy()
    sue_input["SOURCE"] = "REVENUE_MINUS_COGS"
    sue = build_sue_events(sue_input)[
        ["SECURITY_ID", "QUARTER_INDEX", "SUE_RAW", "EVENT_TIME"]
    ].rename(
        columns={"SUE_RAW": "GP_SUE", "EVENT_TIME": "SUE_EVENT_TIME"}
    )
    events = events.merge(
        sue,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="left",
        validate="one_to_one",
    )
    events["EVENT_TIME"] = pd.concat(
        [
            pd.to_datetime(events["EVENT_TIME"], errors="coerce"),
            pd.to_datetime(events["SUE_EVENT_TIME"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    return events.replace([np.inf, -np.inf], np.nan)


def prepare_dense_events(
    events: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    available = assign_available_trade_date(events, calendar)
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
            "GP_GROWTH",
            "GP_ACCELERATION",
            "GP_SUE",
        ]
    ].reset_index(drop=True)


def robust_daily_zscore(
    data: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Daily 1/99 winsorization and MAD z-score; missing becomes neutral zero."""
    values = data[columns].apply(pd.to_numeric, errors="coerce")
    grouped = values.groupby(data["TRADE_DATE"], sort=False)
    lower = grouped.transform(lambda group: group.quantile(0.01))
    upper = grouped.transform(lambda group: group.quantile(0.99))
    clipped = values.clip(lower=lower, upper=upper)
    clipped_grouped = clipped.groupby(data["TRADE_DATE"], sort=False)
    median = clipped_grouped.transform("median")
    absolute_deviation = (clipped - median).abs()
    mad = absolute_deviation.groupby(data["TRADE_DATE"], sort=False).transform(
        "median"
    )
    scale = 1.4826 * mad
    fallback = clipped_grouped.transform("std").replace(0, np.nan)
    scale = scale.where(scale.gt(1e-12), fallback)
    standardized = (clipped - median).div(scale)
    return standardized.clip(-8.0, 8.0).fillna(0.0)


def candidate_specs() -> pd.DataFrame:
    rows: list[dict] = []
    for weight in Q1_WEIGHTS:
        code = int(round(weight * 100))
        rows.extend(
            [
                {
                    "factor": f"grid_gp_level_nodecay_q1w{code:03d}",
                    "version": "gp_level_no_decay",
                    "q1_weight": weight,
                    "half_life": np.nan,
                },
                {
                    "factor": f"grid_gp_acceleration_q1w{code:03d}",
                    "version": "gp_acceleration",
                    "q1_weight": weight,
                    "half_life": np.nan,
                },
            ]
        )
        for half_life in HALF_LIVES:
            rows.extend(
                [
                    {
                        "factor": (
                            f"grid_gp_level_decay_h{half_life}_q1w{code:03d}"
                        ),
                        "version": "gp_level_decay",
                        "q1_weight": weight,
                        "half_life": half_life,
                    },
                    {
                        "factor": (
                            f"grid_gp_sue_decay_h{half_life}_q1w{code:03d}"
                        ),
                        "version": "gp_sue_decay",
                        "q1_weight": weight,
                        "half_life": half_life,
                    },
                ]
            )
    return pd.DataFrame(rows)


def build_grid_for_slice(mapped: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    z = robust_daily_zscore(
        mapped, ["GP_GROWTH", "GP_ACCELERATION", "GP_SUE"]
    )
    q1 = mapped["FISCAL_QUARTER"].eq(1).astype("float64")
    age = pd.to_numeric(mapped["EVENT_AGE"], errors="coerce").clip(lower=0)
    result = mapped[KEYS].copy()
    base_by_version = {
        "gp_level_no_decay": z["GP_GROWTH"],
        "gp_level_decay": z["GP_GROWTH"],
        "gp_acceleration": z["GP_ACCELERATION"],
        "gp_sue_decay": z["GP_SUE"],
    }
    for row in specs.itertuples(index=False):
        value = base_by_version[row.version]
        if pd.notna(row.half_life):
            decay = np.exp(-np.log(2.0) * age / float(row.half_life)).fillna(0.0)
            value = value * decay
        value = value * (1.0 + float(row.q1_weight) * q1)
        result[row.factor] = pd.to_numeric(value, errors="coerce").fillna(0.0).astype(
            "float32"
        )
    return result


def _quality_rows(candidates: pd.DataFrame, specs: pd.DataFrame, year: int) -> pd.DataFrame:
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
                "calendar_days": int(candidates["TRADE_DATE"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def generate_parameter_grid(
    panel_path: Path,
    data_root: Path,
    output_path: Path,
    manifest_path: Path,
    quality_path: Path,
) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique())
    calendar = calendar.normalize().drop_duplicates().sort_values()
    calendar_ordinal = pd.Series(
        np.arange(len(calendar), dtype=np.int16), index=calendar
    )
    events = prepare_dense_events(build_gross_profit_events(data_root), calendar)
    specs = candidate_specs()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    specs.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    quality: list[pd.DataFrame] = []
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
            mapped = map_events_to_panel(panel, events, calendar_ordinal)
            candidates = build_grid_for_slice(mapped, specs)
            year_quality = _quality_rows(candidates, specs, year)
            quality.append(year_quality)
            bad = year_quality.loc[year_quality["constant_days"].gt(0)]
            if not bad.empty:
                examples = bad[["factor", "constant_days"]].to_dict("records")[:5]
                raise RuntimeError(
                    f"{year}存在低方差日期，稠密候选拒绝输出: {examples}"
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
        raise RuntimeError("没有生成稠密参数候选")
    os.replace(temporary, output_path)
    quality_frame = pd.concat(quality, ignore_index=True)
    quality_frame.to_csv(quality_path, index=False, encoding="utf-8-sig")
    print(f"Wrote grid: {output_path}")


def _candidate_training_statistics(
    daily_ic: pd.DataFrame, specs: pd.DataFrame
) -> pd.DataFrame:
    data = daily_ic.copy()
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"])
    data = data.loc[data["TRADE_DATE"].le(TRAIN_END)].copy()
    annual = (
        data.assign(year=data["TRADE_DATE"].dt.year)
        .groupby(["factor", "year"], as_index=False)["neutral_ic"]
        .mean()
    )
    annual_stats = annual.groupby("factor")["neutral_ic"].agg(
        annual_mean="mean",
        annual_std="std",
        positive_years=lambda values: int(values.gt(0).sum()),
        training_years="count",
    )
    daily_stats = data.groupby("factor")["neutral_ic"].agg(
        train_daily_mean="mean",
        train_daily_std="std",
        train_days="count",
    )
    statistics = specs.merge(
        annual_stats, left_on="factor", right_index=True, how="left"
    ).merge(daily_stats, left_on="factor", right_index=True, how="left")
    statistics["train_standard_error"] = statistics["train_daily_std"].div(
        np.sqrt(statistics["train_days"].clip(lower=1))
    )
    statistics["cv_score"] = statistics["annual_mean"] - 0.5 * statistics[
        "annual_std"
    ].fillna(0.0)
    return statistics


def select_parameters(
    daily_ic: pd.DataFrame, specs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select within training only using annual stability and one-SE simplicity."""
    statistics = _candidate_training_statistics(daily_ic, specs)
    selected_rows = []
    for version, group in statistics.groupby("version", sort=False):
        valid = group.loc[
            group["training_years"].ge(5)
            & group["positive_years"].ge(4)
            & group["cv_score"].notna()
        ].copy()
        if valid.empty:
            valid = group.dropna(subset=["cv_score"]).copy()
        best = valid.sort_values("cv_score", ascending=False).iloc[0]
        tolerance = float(best["train_standard_error"])
        within_one_se = valid.loc[
            valid["cv_score"].ge(float(best["cv_score"]) - tolerance)
        ].copy()
        within_one_se["half_life_complexity"] = within_one_se[
            "half_life"
        ].fillna(np.inf)
        chosen = within_one_se.sort_values(
            ["q1_weight", "half_life_complexity", "cv_score"],
            ascending=[True, False, False],
        ).iloc[0]
        selected_rows.append(chosen.drop(labels="half_life_complexity"))
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    dated = daily_ic.copy()
    dated["TRADE_DATE"] = pd.to_datetime(dated["TRADE_DATE"])
    reports = []
    for row in selected.itertuples(index=False):
        values = dated.loc[dated["factor"].eq(row.factor)]
        periods = {
            "train_2017_2022": values["TRADE_DATE"].le(TRAIN_END),
            "validation_2023_2024": values["TRADE_DATE"].between(
                VALIDATION_START, VALIDATION_END
            ),
            "holdout_2025_2026": values["TRADE_DATE"].ge(HOLDOUT_START),
        }
        report = {
            "version": row.version,
            "factor": row.factor,
            "q1_weight": row.q1_weight,
            "half_life": row.half_life,
            "cv_score": row.cv_score,
        }
        for name, mask in periods.items():
            ic = pd.to_numeric(values.loc[mask, "neutral_ic"], errors="coerce").dropna()
            report[f"{name}_ic"] = float(ic.mean()) if len(ic) else np.nan
            report[f"{name}_days"] = len(ic)
        reports.append(report)
    return selected, pd.DataFrame(reports)


def write_selected_factors(
    grid_path: Path,
    selected: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_lookup = dict(zip(selected["version"], selected["factor"]))
    source_columns = [selected_lookup[version] for version in FINAL_NAMES]
    output_names = [FINAL_NAMES[version] for version in FINAL_NAMES]
    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    parquet = pq.ParquetFile(grid_path)
    try:
        for batch in parquet.iter_batches(
            columns=KEYS + source_columns, batch_size=250_000
        ):
            table = pa.Table.from_batches([batch])
            table = table.rename_columns(KEYS + output_names)
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


def fit_and_write(
    daily_ic_path: Path,
    manifest_path: Path,
    grid_path: Path,
    output_path: Path,
    report_path: Path,
    parameters_path: Path,
) -> None:
    daily_ic = pd.read_parquet(daily_ic_path)
    specs = pd.read_csv(manifest_path)
    selected, report = select_parameters(daily_ic, specs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_selected_factors(grid_path, selected, output_path)
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
    print(f"Wrote selected factors: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-grid")
    generate.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    generate.add_argument(
        "--data-root", type=Path, default=BASE_DIR / "data" / "new_pit"
    )
    generate.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_q1_parameter_grid.parquet",
    )
    generate.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_q1_parameter_grid.csv",
    )
    generate.add_argument(
        "--quality",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_q1_grid_quality.csv",
    )
    fit = subparsers.add_parser("fit")
    fit.add_argument("--daily-ic", type=Path, required=True)
    fit.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_q1_parameter_grid.csv",
    )
    fit.add_argument(
        "--grid",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_q1_parameter_grid.parquet",
    )
    fit.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_q1_gross_profit_factors.parquet",
    )
    fit.add_argument(
        "--report",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_q1_parameter_report.csv",
    )
    fit.add_argument(
        "--parameters",
        type=Path,
        default=BASE_DIR / "factor_components" / "dense_q1_selected_parameters.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-grid":
        generate_parameter_grid(
            args.panel.resolve(),
            args.data_root.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
            args.quality.resolve(),
        )
    else:
        fit_and_write(
            args.daily_ic.resolve(),
            args.manifest.resolve(),
            args.grid.resolve(),
            args.output.resolve(),
            args.report.resolve(),
            args.parameters.resolve(),
        )


if __name__ == "__main__":
    main()
