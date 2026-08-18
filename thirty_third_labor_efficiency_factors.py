"""Build PIT-safe annual labor-efficiency factors without percentile ranks.

The source table ``fdmt_avg_sal_empl`` supplies annual staff count and total
employee compensation.  Annual revenue/profit are taken from the new-standard
PIT income statement.  Signals become available only after both reports have
been published.  Missing company histories are assigned the neutral value 0;
this is appropriate for point-in-time state data and preserves the complete
daily test universe required by ``factors_neus_only2.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS, _normalize_common
from twenty_third_alternative_event_factors import KEYS, map_party, security_mapping


FACTOR_COLUMNS = [
    "r33_sales_per_employee_growth",
    "r33_profit_growth_less_staff_growth",
    "r33_sales_salary_efficiency_growth",
    "r33_low_staff_growth",
]


def read_partitioned(root: Path, columns: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.parquet")):
        names = pq.read_schema(path).names
        if all(column in names for column in columns):
            pieces.append(pd.read_parquet(path, columns=columns))
    if not pieces:
        raise FileNotFoundError(f"No compatible parquet files below {root}")
    return pd.concat(pieces, ignore_index=True)


def safe_growth(current: pd.Series, prior: pd.Series) -> pd.Series:
    floor = prior.abs().where(prior.abs().gt(1e-9))
    return ((current - prior) / floor).clip(-5.0, 5.0)


def annual_income(pit_root: Path) -> pd.DataFrame:
    columns = [*COMMON_COLUMNS, "REVENUE", "N_INCOME_ATTR_P"]
    raw = read_partitioned(pit_root / "new_pit_income", columns)
    revenue = _normalize_common(raw, "REVENUE", name="annual revenue")
    profit = _normalize_common(raw, "N_INCOME_ATTR_P", name="annual profit")
    keep = ["SECURITY_ID", "END_DATE", "REPORT_TYPE", "FISCAL_PERIOD"]
    revenue = revenue.loc[
        revenue["REPORT_TYPE"].eq("A") & revenue["FISCAL_PERIOD"].eq(12),
        keep + ["ACT_PUBTIME", "REVENUE"],
    ].rename(columns={"ACT_PUBTIME": "REVENUE_TIME"})
    profit = profit.loc[
        profit["REPORT_TYPE"].eq("A") & profit["FISCAL_PERIOD"].eq(12),
        keep + ["ACT_PUBTIME", "N_INCOME_ATTR_P"],
    ].rename(columns={"ACT_PUBTIME": "PROFIT_TIME"})
    data = revenue.merge(profit, on=keep, how="inner", validate="one_to_one")
    data["INCOME_TIME"] = data[["REVENUE_TIME", "PROFIT_TIME"]].max(axis=1)
    return data[["SECURITY_ID", "END_DATE", "INCOME_TIME", "REVENUE", "N_INCOME_ATTR_P"]]


def labor_reports(alternative_root: Path, pit_root: Path) -> pd.DataFrame:
    columns = [
        "PARTY_ID", "TICKER_SYMBOL", "PUBLISH_DATE", "END_DATE",
        "REPORT_TYPE", "MERGED_FLAG", "SUM_SAL_EMPL", "STAFF_NUM",
    ]
    raw = read_partitioned(alternative_root / "fdmt_avg_sal_empl", columns)
    pair, _ = security_mapping(pit_root)
    raw = map_party(raw, pair)
    raw["EVENT_TIME_LABOR"] = (
        pd.to_datetime(raw["PUBLISH_DATE"], errors="coerce").dt.normalize()
        + pd.Timedelta(1439, unit="m")
    )
    raw["END_DATE"] = pd.to_datetime(raw["END_DATE"], errors="coerce").dt.normalize()
    for column in ["SUM_SAL_EMPL", "STAFF_NUM"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.loc[
        raw["REPORT_TYPE"].astype(str).eq("A")
        & raw["MERGED_FLAG"].astype(str).eq("1")
        & raw["STAFF_NUM"].gt(0)
        & raw["SUM_SAL_EMPL"].gt(0)
    ].dropna(subset=["SECURITY_ID", "END_DATE", "EVENT_TIME_LABOR"])
    return (
        raw.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME_LABOR"])
        .drop_duplicates(["SECURITY_ID", "END_DATE"], keep="first")
        [["SECURITY_ID", "END_DATE", "EVENT_TIME_LABOR", "SUM_SAL_EMPL", "STAFF_NUM"]]
    )


def build_events(labor: pd.DataFrame, income: pd.DataFrame) -> pd.DataFrame:
    data = labor.merge(income, on=["SECURITY_ID", "END_DATE"], how="inner", validate="one_to_one")
    data["EVENT_TIME"] = data[["EVENT_TIME_LABOR", "INCOME_TIME"]].max(axis=1)
    data = data.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME"])
    for column in ["REVENUE", "N_INCOME_ATTR_P", "SUM_SAL_EMPL", "STAFF_NUM"]:
        data[f"L1_{column}"] = data.groupby("SECURITY_ID")[column].shift(1)
    consecutive = data["END_DATE"].dt.year.sub(
        data.groupby("SECURITY_ID")["END_DATE"].shift(1).dt.year
    ).eq(1)
    staff_growth = safe_growth(data["STAFF_NUM"], data["L1_STAFF_NUM"])
    revenue_growth = safe_growth(data["REVENUE"], data["L1_REVENUE"])
    profit_growth = safe_growth(data["N_INCOME_ATTR_P"], data["L1_N_INCOME_ATTR_P"])
    sales_per_employee = np.log(data["REVENUE"].clip(lower=1.0) / data["STAFF_NUM"])
    prior_sales_per_employee = np.log(
        data["L1_REVENUE"].clip(lower=1.0) / data["L1_STAFF_NUM"]
    )
    sales_salary = np.log(data["REVENUE"].clip(lower=1.0) / data["SUM_SAL_EMPL"])
    prior_sales_salary = np.log(
        data["L1_REVENUE"].clip(lower=1.0) / data["L1_SUM_SAL_EMPL"]
    )
    data["r33_sales_per_employee_growth"] = (sales_per_employee - prior_sales_per_employee).clip(-3, 3)
    data["r33_profit_growth_less_staff_growth"] = (profit_growth - staff_growth).clip(-5, 5)
    data["r33_sales_salary_efficiency_growth"] = (sales_salary - prior_sales_salary).clip(-3, 3)
    data["r33_low_staff_growth"] = (-staff_growth).clip(-5, 5)
    data.loc[~consecutive, FACTOR_COLUMNS] = np.nan
    return data.dropna(subset=FACTOR_COLUMNS, how="all")[["SECURITY_ID", "END_DATE", "EVENT_TIME", *FACTOR_COLUMNS]]


def dense_state(panel: pd.DataFrame, events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    available = assign_available_trade_date(events, calendar).rename(columns={"AVAILABLE_DATE": "EVENT_DATE"})
    available = (
        available.sort_values(["EVENT_DATE", "SECURITY_ID", "EVENT_TIME"])
        .groupby(["EVENT_DATE", "SECURITY_ID"], as_index=False)[FACTOR_COLUMNS]
        .last()
    )
    left = panel.sort_values(["TRADE_DATE", "SECURITY_ID"])
    right = available.sort_values(["EVENT_DATE", "SECURITY_ID"])
    merged = pd.merge_asof(
        left, right, left_on="TRADE_DATE", right_on="EVENT_DATE",
        by="SECURITY_ID", direction="backward", allow_exact_matches=True,
    )
    result = merged[KEYS + FACTOR_COLUMNS].copy()
    # Point-in-time annual states may be carried forward.  Companies without
    # a two-year history receive the economically neutral cross-sectional value.
    result[FACTOR_COLUMNS] = result[FACTOR_COLUMNS].fillna(0.0).astype("float32")
    return result.sort_values(KEYS).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternative-root", type=Path, required=True)
    parser.add_argument("--pit-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel, columns=KEYS).drop_duplicates(KEYS)
    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
    panel["SECURITY_ID"] = panel["SECURITY_ID"].astype("int64")
    calendar = pd.DatetimeIndex(panel["TRADE_DATE"].unique()).sort_values()
    labor = labor_reports(args.alternative_root, args.pit_root)
    income = annual_income(args.pit_root)
    events = build_events(labor, income)
    factors = dense_state(panel, events, calendar)
    events.to_parquet(output / "round33_labor_efficiency_events.parquet", index=False)
    factors.to_parquet(output / "round33_labor_efficiency_factors.parquet", index=False)
    diagnostics = {
        "event_rows": len(events),
        "event_securities": int(events["SECURITY_ID"].nunique()),
        "event_date_min": str(events["EVENT_TIME"].min()),
        "event_date_max": str(events["EVENT_TIME"].max()),
        "factor_columns": FACTOR_COLUMNS,
        "event_non_null": {column: int(events[column].notna().sum()) for column in FACTOR_COLUMNS},
    }
    (output / "metadata.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
