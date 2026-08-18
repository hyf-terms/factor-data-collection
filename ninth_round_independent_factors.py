"""Independent financing, capital-structure and cash factors for round nine.

Period-flow candidates follow a mandatory two-stage workflow: generate and
test the sparse PIT panel first, then permit same-date neutral filling.  Stock
variables from the balance sheet are point-in-time observations and persist
until a newer public report becomes available.  No percentile rank and no
blend with any existing best factor is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import (
    COMMON_COLUMNS,
    FINANCIAL_INDUSTRIES,
    REPORT_QUARTERS,
    REPORT_TYPES,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent

CASHFLOW_FIELDS = [
    "N_CF_FR_FINAN_A",
    "C_FR_CAP_CONTR",
    "C_FR_BORR",
    "C_FR_ISSUE_BOND",
    "C_PAID_FOR_DEBTS",
    "C_PAID_DIV_PROF_INT",
]

BALANCE_FIELDS = [
    "INDUSTRY_CATEGORY",
    "T_ASSETS",
    "CASH_C_EQUIV",
    "ST_BORR",
    "NCL_WITHIN_1_Y",
    "LT_BORR",
    "BOND_PAYABLE",
    "PAID_IN_CAPITAL",
    "T_EQUITY_ATTR_P",
    "RETAINED_EARNINGS",
    "INTAN_ASSETS",
    "GOODWILL",
]

# Missing component stocks mean the item is absent from that point-in-time
# balance sheet.  This is distinct from filling an unavailable period flow.
ZERO_BALANCE_COMPONENTS = {
    "ST_BORR",
    "NCL_WITHIN_1_Y",
    "LT_BORR",
    "BOND_PAYABLE",
    "INTAN_ASSETS",
    "GOODWILL",
}

CANDIDATE_COLUMNS = [
    "r9_brs_q_low_net_external_financing",
    "r9_brs_ttm_low_net_external_financing",
    "r9_q_low_net_debt_financing",
    "r9_ttm_low_net_debt_financing",
    "r9_q_low_equity_financing",
    "r9_ttm_low_equity_financing",
    "r9_q_financing_distribution_assets",
    "r9_low_share_issuance",
    "r9_low_debt_growth",
    "r9_cash_assets",
    "r9_cash_accumulation",
    "r9_low_intangible_growth",
    "r9_retained_earnings_assets",
    "r9_low_equity_growth",
    "r9_low_leverage_change",
]


def _latest_time(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.concat(
        [pd.to_datetime(frame[column], errors="coerce") for column in columns],
        axis=1,
    ).max(axis=1)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    scale = denominator.abs().median(skipna=True)
    floor = max(float(scale) * 1e-8, 1e-12) if pd.notna(scale) else 1e-12
    valid = denominator.abs().gt(floor) & np.isfinite(denominator)
    return (
        pd.to_numeric(numerator, errors="coerce")
        .div(denominator.where(valid))
        .replace([np.inf, -np.inf], np.nan)
    )


def _read_inputs(pit_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    cashflow = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=COMMON_COLUMNS + CASHFLOW_FIELDS,
        engine="pyarrow",
    )
    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    flow_tables = {
        field: build_standalone_quarterly_metric(
            cashflow, field, name="现金流量表PIT"
        )
        for field in CASHFLOW_FIELDS
    }
    return flow_tables, build_balance_events(balance)


def build_balance_events(balance: pd.DataFrame) -> pd.DataFrame:
    data = balance[COMMON_COLUMNS + BALANCE_FIELDS].copy()
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce")
    for field in BALANCE_FIELDS:
        if field != "INDUSTRY_CATEGORY":
            data[field] = pd.to_numeric(data[field], errors="coerce")
    for field in ZERO_BALANCE_COMPONENTS:
        data[field] = data[field].fillna(0.0)
    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & data["END_DATE"].eq(data["END_DATE_REP"])
        & ~data["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES)
    )
    data = data.loc[mask].dropna(
        subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "T_ASSETS"]
    )
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")
    data["FISCAL_QUARTER"] = data["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    data["QUARTER_INDEX"] = data["FISCAL_YEAR"].astype("int64") * 4 + data["FISCAL_QUARTER"]
    key = ["SECURITY_ID", "QUARTER_INDEX"]
    data = data.sort_values(key + ["ACT_PUBTIME", "ID"]).drop_duplicates(key, keep="first")
    return data[
        key + ["FISCAL_YEAR", "FISCAL_QUARTER", "END_DATE", "ACT_PUBTIME", *BALANCE_FIELDS]
    ].rename(columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"})


def _lag_table(
    frame: pd.DataFrame,
    quarters: int,
    prefix: str,
    columns: list[str],
) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "QUARTER_INDEX", *columns]].copy()
    result["QUARTER_INDEX"] += quarters
    return result.rename(columns={column: f"{prefix}_{column}" for column in columns})


def _long_event(
    frame: pd.DataFrame,
    factor: str,
    values: pd.Series,
    event_columns: list[str],
) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    result["EVENT_TIME"] = _latest_time(frame, event_columns)
    result["factor"] = factor
    result["value"] = pd.to_numeric(values, errors="coerce")
    return result.dropna(subset=["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", "value"])


def _flow_with_balance(
    flow_tables: dict[str, pd.DataFrame],
    fields: list[str],
    balance: pd.DataFrame,
    *,
    ttm: bool,
) -> pd.DataFrame:
    flows = _merge_flow_tables(flow_tables, fields, ttm=ttm)
    lag = 4 if ttm else 1
    prefix = "L4" if ttm else "L1"
    lagged_balance = _lag_table(
        balance, lag, prefix, ["BALANCE_EVENT_TIME", "T_ASSETS"]
    )
    return flows.merge(
        lagged_balance,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )


def calculate_factor_events(
    flow_tables: dict[str, pd.DataFrame], balance: pd.DataFrame
) -> pd.DataFrame:
    events: list[pd.DataFrame] = []

    for ttm, suffix, prefix in (
        (False, "q", "L1"),
        (True, "ttm", "L4"),
    ):
        event_columns = ["FLOW_EVENT_TIME", f"{prefix}_BALANCE_EVENT_TIME"]
        net = _flow_with_balance(
            flow_tables, ["N_CF_FR_FINAN_A"], balance, ttm=ttm
        )
        value_column = "TTM_N_CF_FR_FINAN_A" if ttm else "N_CF_FR_FINAN_A"
        events.append(
            _long_event(
                net,
                f"r9_brs_{suffix}_low_net_external_financing",
                -_safe_ratio(net[value_column], net[f"{prefix}_T_ASSETS"]),
                event_columns,
            )
        )

        debt_fields = ["C_FR_BORR", "C_FR_ISSUE_BOND", "C_PAID_FOR_DEBTS"]
        debt = _flow_with_balance(flow_tables, debt_fields, balance, ttm=ttm)
        name = lambda field: f"TTM_{field}" if ttm else field
        net_debt = (
            debt[name("C_FR_BORR")]
            + debt[name("C_FR_ISSUE_BOND")]
            - debt[name("C_PAID_FOR_DEBTS")]
        )
        events.append(
            _long_event(
                debt,
                f"r9_{suffix}_low_net_debt_financing",
                -_safe_ratio(net_debt, debt[f"{prefix}_T_ASSETS"]),
                event_columns,
            )
        )

        equity = _flow_with_balance(
            flow_tables, ["C_FR_CAP_CONTR"], balance, ttm=ttm
        )
        events.append(
            _long_event(
                equity,
                f"r9_{suffix}_low_equity_financing",
                -_safe_ratio(
                    equity[name("C_FR_CAP_CONTR")],
                    equity[f"{prefix}_T_ASSETS"],
                ),
                event_columns,
            )
        )

        if not ttm:
            distribution = _flow_with_balance(
                flow_tables, ["C_PAID_DIV_PROF_INT"], balance, ttm=False
            )
            events.append(
                _long_event(
                    distribution,
                    "r9_q_financing_distribution_assets",
                    _safe_ratio(
                        distribution["C_PAID_DIV_PROF_INT"],
                        distribution["L1_T_ASSETS"],
                    ),
                    ["FLOW_EVENT_TIME", "L1_BALANCE_EVENT_TIME"],
                )
            )

    lag_columns = [
        "BALANCE_EVENT_TIME",
        *[field for field in BALANCE_FIELDS if field != "INDUSTRY_CATEGORY"],
    ]
    l4 = _lag_table(balance, 4, "L4", lag_columns)
    data = balance.merge(
        l4,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    times = ["BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]

    def debt(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
        marker = f"{prefix}_" if prefix else ""
        return sum(
            frame[f"{marker}{field}"]
            for field in ("ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE")
        )

    current_debt = debt(data)
    prior_debt = debt(data, "L4")
    current_intangible = data["INTAN_ASSETS"] + data["GOODWILL"]
    prior_intangible = data["L4_INTAN_ASSETS"] + data["L4_GOODWILL"]
    current_leverage = _safe_ratio(current_debt, data["T_EQUITY_ATTR_P"])
    prior_leverage = _safe_ratio(prior_debt, data["L4_T_EQUITY_ATTR_P"])

    balance_values = {
        "r9_low_share_issuance": -(
            _safe_ratio(data["PAID_IN_CAPITAL"], data["L4_PAID_IN_CAPITAL"]) - 1.0
        ),
        "r9_low_debt_growth": -_safe_ratio(
            current_debt - prior_debt, data["L4_T_ASSETS"]
        ),
        "r9_cash_assets": _safe_ratio(data["CASH_C_EQUIV"], data["T_ASSETS"]),
        "r9_cash_accumulation": _safe_ratio(
            data["CASH_C_EQUIV"] - data["L4_CASH_C_EQUIV"],
            data["L4_T_ASSETS"],
        ),
        "r9_low_intangible_growth": -_safe_ratio(
            current_intangible - prior_intangible, data["L4_T_ASSETS"]
        ),
        "r9_retained_earnings_assets": _safe_ratio(
            data["RETAINED_EARNINGS"], data["T_ASSETS"]
        ),
        "r9_low_equity_growth": -(
            _safe_ratio(data["T_EQUITY_ATTR_P"], data["L4_T_EQUITY_ATTR_P"]) - 1.0
        ),
        "r9_low_leverage_change": -(current_leverage - prior_leverage),
    }
    events.extend(
        _long_event(data, factor, value, times)
        for factor, value in balance_values.items()
    )

    result = pd.concat(events, ignore_index=True)
    result["QUARTER_INDEX"] = pd.to_numeric(result["QUARTER_INDEX"]).astype("int64")
    return result.sort_values(
        ["factor", "SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"]
    ).reset_index(drop=True)


def prepare_wide_events(events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for factor, group in events.groupby("factor", sort=False):
        available = assign_available_trade_date(group, calendar)
        available = available.sort_values(
            ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
        )
        newest = available.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
        available = available.loc[available["QUARTER_INDEX"].eq(newest)]
        available = available.drop_duplicates(
            ["SECURITY_ID", "AVAILABLE_DATE"], keep="last"
        )
        available["factor"] = factor
        pieces.append(
            available[["SECURITY_ID", "AVAILABLE_DATE", "factor", "value"]]
        )
    long = pd.concat(pieces, ignore_index=True)
    wide = long.pivot_table(
        index=["SECURITY_ID", "AVAILABLE_DATE"],
        columns="factor",
        values="value",
        aggfunc="last",
    ).reset_index()
    for factor in CANDIDATE_COLUMNS:
        if factor not in wide:
            wide[factor] = np.nan
    wide = wide.sort_values(["SECURITY_ID", "AVAILABLE_DATE"])
    wide[CANDIDATE_COLUMNS] = wide.groupby(
        "SECURITY_ID", sort=False
    )[CANDIDATE_COLUMNS].ffill()
    return wide[["SECURITY_ID", "AVAILABLE_DATE", *CANDIDATE_COLUMNS]]


def _map_sparse(panel: pd.DataFrame, wide_events: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.merge_asof(
        panel.sort_values(["TRADE_DATE", "SECURITY_ID"]),
        wide_events.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]),
        by="SECURITY_ID",
        left_on="TRADE_DATE",
        right_on="AVAILABLE_DATE",
        direction="backward",
    )
    for factor in CANDIDATE_COLUMNS:
        mapped[factor] = pd.to_numeric(mapped[factor], errors="coerce").astype("float32")
    return mapped[KEYS + CANDIDATE_COLUMNS].sort_values(KEYS)


def generate_sparse(
    panel_path: Path,
    pit_dir: Path,
    output: Path,
    coverage_output: Path,
    metadata_output: Path,
) -> None:
    panel_dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(panel_dates).unique()).normalize().sort_values()
    flows, balance = _read_inputs(pit_dir)
    events = calculate_factor_events(flows, balance)
    wide = prepare_wide_events(events, calendar)
    chunks: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for year in sorted(set(calendar.year)):
        filters = [
            ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
            ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
        ]
        panel = _normalize_panel(
            pd.read_parquet(panel_path, columns=KEYS, filters=filters)
        )
        mapped = _map_sparse(panel, wide)
        chunks.append(mapped)
        for factor in CANDIDATE_COLUMNS:
            values = mapped[factor]
            coverage_rows.append(
                {
                    "year": year,
                    "factor": factor,
                    "rows": len(mapped),
                    "observed_rows": int(values.notna().sum()),
                    "missing_rate_before_fill": float(values.isna().mean()),
                    "observed_days": int(
                        mapped.loc[values.notna(), "TRADE_DATE"].nunique()
                    ),
                }
            )
        print(f"{year}: sparse rows={len(mapped):,}")
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    pd.DataFrame(coverage_rows).to_csv(
        coverage_output, index=False, encoding="utf-8-sig"
    )
    metadata_output.write_text(
        json.dumps(
            {
                "stage": "sparse_before_fill",
                "rows": len(result),
                "factors": CANDIDATE_COLUMNS,
                "period_source_zero_fill": False,
                "balance_absent_component_zero": sorted(ZERO_BALANCE_COMPONENTS),
                "daily_cross_section_fill": False,
                "existing_factor_blend": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def fill_after_test(
    sparse_path: Path,
    sparse_ic_summary: Path,
    output: Path,
    report_output: Path,
    factor_columns: list[str] | None = None,
) -> None:
    if not sparse_ic_summary.exists():
        raise FileNotFoundError(
            f"必须先完成未填充版本测试，缺少: {sparse_ic_summary}"
        )
    tested = pd.read_csv(sparse_ic_summary)
    selected = factor_columns or CANDIDATE_COLUMNS
    unknown = sorted(set(selected).difference(CANDIDATE_COLUMNS))
    if unknown:
        raise KeyError(f"未知第九轮因子: {unknown}")
    missing_test = sorted(
        set(selected).difference(tested["factor"].astype(str))
    )
    if missing_test:
        raise RuntimeError(f"未填充测试缺少因子: {missing_test}")
    data = pd.read_parquet(sparse_path)
    before = data[selected].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[selected].transform(
        "median"
    )
    data[selected] = data[selected].fillna(medians)
    remaining = data[selected].isna().sum()
    if remaining.any():
        raise RuntimeError(
            "当日中位数仍无法填充: "
            + str(remaining.loc[remaining.gt(0)].to_dict())
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    data[KEYS + selected].to_parquet(output, index=False, compression="zstd")
    pd.DataFrame(
        {
            "factor": selected,
            "missing_rate_before_fill": [float(before[f]) for f in selected],
            "fill_method": "same_date_cross_section_median_after_sparse_test",
            "remaining_missing_rows": [int(remaining[f]) for f in selected],
        }
    ).to_csv(report_output, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    root = BASE_DIR / "新测试结果" / "第九轮低相关独立因子"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    generate.add_argument("--pit-dir", type=Path, default=BASE_DIR / "data" / "new_pit")
    generate.add_argument("--output", type=Path, default=root / "round9_sparse_before_fill.parquet")
    generate.add_argument("--coverage", type=Path, default=root / "round9_sparse_coverage.csv")
    generate.add_argument("--metadata", type=Path, default=root / "round9_sparse_metadata.json")
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, default=root / "round9_sparse_before_fill.parquet")
    fill.add_argument("--sparse-ic", type=Path, default=root / "sparse_diagnostic_test" / "ic_summary.csv")
    fill.add_argument("--output", type=Path, default=root / "round9_filled_after_test.parquet")
    fill.add_argument("--report", type=Path, default=root / "round9_fill_report.csv")
    fill.add_argument("--factor-columns", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-sparse":
        generate_sparse(
            args.panel.resolve(),
            args.pit_dir.resolve(),
            args.output.resolve(),
            args.coverage.resolve(),
            args.metadata.resolve(),
        )
    else:
        fill_after_test(
            args.sparse.resolve(),
            args.sparse_ic.resolve(),
            args.output.resolve(),
            args.report.resolve(),
            args.factor_columns,
        )


if __name__ == "__main__":
    main()
