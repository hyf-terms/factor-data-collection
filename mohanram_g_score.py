"""Build a PIT-safe quarterly Mohanram G-score.

The original Mohanram (2005) score is designed for low book-to-market growth
firms.  This implementation uses trailing-four-quarter fundamentals, quarterly
history for the two stability signals, and the daily Barra book-to-price and
industry exposures to create point-in-time peer benchmarks.

The PIT income statements do not contain a separate advertising-expense field.
``SELL_EXP`` is therefore used as an explicit selling/advertising proxy for G8.
The proxy is disclosed in the audit output and documentation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import (
    COMMON_COLUMNS,
    build_quarterly_balance,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PIT_DIR = BASE_DIR / "data" / "new_pit"
DEFAULT_FACTOR_PATH = BASE_DIR / "factors.parquet"
DEFAULT_BARRA_PATH = BASE_DIR / "barra_diy.parquet"
DEFAULT_AUDIT_DIR = BASE_DIR / "factor_components"

FACTOR_NAME = "mohanram_g_score"
KEYS = ["TRADE_DATE", "SECURITY_ID"]
FLOW_COLUMNS = [
    "N_INCOME_ATTR_P",
    "N_CF_OPERATE_A",
    "REVENUE",
    "R_D_EXP",
    "PUR_FIX_ASSETS_OTH",
    "SELL_EXP",
]
INCOME_VALUE_COLUMNS = [
    "N_INCOME_ATTR_P",
    "REVENUE",
    "R_D_EXP",
    "SELL_EXP",
]
CASHFLOW_VALUE_COLUMNS = ["N_CF_OPERATE_A", "PUR_FIX_ASSETS_OTH"]
ZERO_IF_UNDISCLOSED = {"R_D_EXP", "SELL_EXP", "PUR_FIX_ASSETS_OTH"}
PEER_METRICS = [
    "ROA",
    "CFROA",
    "VAR_ROA",
    "VAR_SALES_GROWTH",
    "RD_INTENSITY",
    "CAPEX_INTENSITY",
    "AD_PROXY_INTENSITY",
]
SIGNAL_COLUMNS = [
    "G1_ROA",
    "G2_CFROA",
    "G3_CFO_ABOVE_INCOME",
    "G4_EARNINGS_STABILITY",
    "G5_GROWTH_STABILITY",
    "G6_RD_INTENSITY",
    "G7_CAPEX_INTENSITY",
    "G8_AD_PROXY_INTENSITY",
]
STYLE_COLUMNS = [
    "liquidity",
    "leverage",
    "earnings_variability",
    "earnings_quality",
    "profitability",
    "investment_quality",
    "book_to_price",
    "earnings_yield",
    "longterm_reversal",
    "growth",
    "momentum",
    "mid_cap",
    "size",
    "beta",
    "residual_volatility",
    "dividend_yield",
    "industry_momentum",
    "sentiment",
    "seasonality",
    "shortterm_reversal",
]
MIN_INDUSTRY_FIRMS = 4
GROWTH_QUANTILE = 0.20
HISTORY_QUARTERS = 16
MIN_HISTORY_OBSERVATIONS = 6


def _latest_time(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.concat(
        [pd.to_datetime(frame[column], errors="coerce") for column in columns],
        axis=1,
    ).max(axis=1)


def _read_and_prepare_statements(
    pit_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    income_columns = COMMON_COLUMNS + INCOME_VALUE_COLUMNS
    cashflow_columns = COMMON_COLUMNS + CASHFLOW_VALUE_COLUMNS
    income = pd.read_parquet(
        pit_dir / "new_pit_income",
        columns=income_columns,
        engine="pyarrow",
    )
    cashflow = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=cashflow_columns,
        engine="pyarrow",
    )

    # In financial databases an undisclosed R&D/advertising/capex item is
    # conventionally treated as zero for the conservatism signals.  We retain
    # the report's actual publication time, rather than inventing an earlier
    # observation.
    for column in ZERO_IF_UNDISCLOSED.intersection(income.columns):
        income[column] = pd.to_numeric(
            income[column], errors="coerce"
        ).fillna(0.0)
    for column in ZERO_IF_UNDISCLOSED.intersection(cashflow.columns):
        cashflow[column] = pd.to_numeric(
            cashflow[column], errors="coerce"
        ).fillna(0.0)

    from quarterly_f_score import BALANCE_COLUMNS

    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=BALANCE_COLUMNS,
        engine="pyarrow",
    )
    return income, cashflow, balance


def build_quarterly_flows(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> pd.DataFrame:
    """Align standalone-quarter values and their first availability times."""
    frames: list[tuple[str, pd.DataFrame]] = []
    for column in INCOME_VALUE_COLUMNS:
        frames.append(
            (
                column,
                build_standalone_quarterly_metric(
                    income,
                    column,
                    name="利润表PIT",
                ),
            )
        )
    for column in CASHFLOW_VALUE_COLUMNS:
        frames.append(
            (
                column,
                build_standalone_quarterly_metric(
                    cashflow,
                    column,
                    name="现金流量表PIT",
                ),
            )
        )

    keys = [
        "SECURITY_ID",
        "FISCAL_YEAR",
        "FISCAL_QUARTER",
        "QUARTER_INDEX",
    ]
    first_column, first = frames[0]
    base = first[
        keys + ["END_DATE", "EVENT_TIME", first_column]
    ].rename(columns={"EVENT_TIME": f"{first_column}_EVENT_TIME"})
    for column, frame in frames[1:]:
        other = frame[
            keys + ["EVENT_TIME", column]
        ].rename(columns={"EVENT_TIME": f"{column}_EVENT_TIME"})
        base = pd.merge(
            base,
            other,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    event_columns = [f"{column}_EVENT_TIME" for column in FLOW_COLUMNS]
    base["FLOW_EVENT_TIME"] = _latest_time(base, event_columns)
    return base.sort_values(["SECURITY_ID", "QUARTER_INDEX"]).reset_index(
        drop=True
    )


def _rolling_metric(
    group: pd.DataFrame,
    value_column: str,
    event_column: str,
    *,
    window: int,
    minimum: int,
    operation: str,
) -> pd.DataFrame:
    indexed = group.set_index("QUARTER_INDEX").sort_index()
    full_index = pd.RangeIndex(
        int(indexed.index.min()),
        int(indexed.index.max()) + 1,
    )
    full = indexed.reindex(full_index)
    values = pd.to_numeric(full[value_column], errors="coerce")
    roller = values.rolling(window, min_periods=minimum)
    if operation == "sum":
        calculated = roller.sum()
    elif operation == "var":
        calculated = roller.var(ddof=1)
    else:
        raise ValueError(f"不支持的滚动操作: {operation}")

    times = pd.to_datetime(full[event_column], errors="coerce")
    time_ns = pd.Series(
        times.astype("int64").to_numpy(dtype=np.float64),
        index=full_index,
    ).where(times.notna())
    available_ns = time_ns.rolling(
        window, min_periods=minimum
    ).max()
    available = pd.to_datetime(available_ns, unit="ns", errors="coerce")
    return pd.DataFrame(
        {
            "SECURITY_ID": int(group["SECURITY_ID"].iloc[0]),
            "QUARTER_INDEX": full_index,
            value_column: calculated.to_numpy(),
            f"{value_column}_AVAILABLE_TIME": available.to_numpy(),
        }
    )


def _build_ttm_flows(flows: pd.DataFrame) -> pd.DataFrame:
    result = flows.sort_values(
        ["SECURITY_ID", "QUARTER_INDEX"]
    ).reset_index(drop=True)
    # A new segment prevents a four-observation rolling sum from jumping over
    # a missing fiscal quarter.
    discontinuity = (
        result.groupby("SECURITY_ID")["QUARTER_INDEX"].diff().ne(1)
    )
    result["_SEGMENT"] = discontinuity.groupby(
        result["SECURITY_ID"]
    ).cumsum()
    group_keys = [result["SECURITY_ID"], result["_SEGMENT"]]
    for column in FLOW_COLUMNS:
        rolled_value = (
            result.groupby(group_keys, sort=False)[column]
            .rolling(4, min_periods=4)
            .sum()
            .reset_index(level=[0, 1], drop=True)
            .reindex(result.index)
        )
        result[f"TTM_{column}"] = rolled_value
        times = pd.to_datetime(
            result[f"{column}_EVENT_TIME"], errors="coerce"
        )
        time_ns = pd.Series(
            times.astype("int64").to_numpy(dtype=np.float64),
            index=result.index,
        ).where(times.notna())
        available_ns = (
            time_ns.groupby(group_keys, sort=False)
            .rolling(4, min_periods=4)
            .max()
            .reset_index(level=[0, 1], drop=True)
            .reindex(result.index)
        )
        result[f"{column}_AVAILABLE_TIME"] = pd.to_datetime(
            available_ns, unit="ns", errors="coerce"
        )
    time_columns = [
        f"{column}_AVAILABLE_TIME" for column in FLOW_COLUMNS
    ]
    result["TTM_EVENT_TIME"] = _latest_time(result, time_columns)
    result["FISCAL_YEAR"] = (
        (result["QUARTER_INDEX"] - 1) // 4
    ).astype("int16")
    result["FISCAL_QUARTER"] = (
        (result["QUARTER_INDEX"] - 1) % 4 + 1
    ).astype("int8")
    return result.drop(columns="_SEGMENT").dropna(
        subset=[f"TTM_{column}" for column in FLOW_COLUMNS]
        + ["TTM_EVENT_TIME"]
    )


def _build_quarterly_history(
    flows: pd.DataFrame,
    balances: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate 16-quarter ROA and year-over-year sales-growth variance."""
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    selected_flows = flows[
        keys
        + [
            "N_INCOME_ATTR_P",
            "N_INCOME_ATTR_P_EVENT_TIME",
            "REVENUE",
            "REVENUE_EVENT_TIME",
        ]
    ]
    current_balance = balances[
        keys + ["T_ASSETS", "BALANCE_EVENT_TIME"]
    ]
    previous_balance = current_balance.copy()
    previous_balance["QUARTER_INDEX"] += 1
    previous_balance = previous_balance.rename(
        columns={
            "T_ASSETS": "PREVIOUS_ASSETS",
            "BALANCE_EVENT_TIME": "PREVIOUS_ASSETS_EVENT_TIME",
        }
    )
    history = pd.merge(
        selected_flows,
        current_balance,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    history = pd.merge(
        history,
        previous_balance,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    average_assets = (history["T_ASSETS"] + history["PREVIOUS_ASSETS"]) / 2
    history["QUARTERLY_ROA"] = history["N_INCOME_ATTR_P"].div(
        average_assets
    ).where(average_assets.gt(0))
    history["QUARTERLY_ROA_EVENT_TIME"] = _latest_time(
        history,
        [
            "N_INCOME_ATTR_P_EVENT_TIME",
            "BALANCE_EVENT_TIME",
            "PREVIOUS_ASSETS_EVENT_TIME",
        ],
    )

    lagged_sales = selected_flows[
        keys + ["REVENUE", "REVENUE_EVENT_TIME"]
    ].copy()
    lagged_sales["QUARTER_INDEX"] += 4
    lagged_sales = lagged_sales.rename(
        columns={
            "REVENUE": "LAG4_REVENUE",
            "REVENUE_EVENT_TIME": "LAG4_REVENUE_EVENT_TIME",
        }
    )
    history = pd.merge(
        history,
        lagged_sales,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    history["SALES_GROWTH"] = (
        history["REVENUE"].div(history["LAG4_REVENUE"]) - 1
    ).where(history["LAG4_REVENUE"].gt(0))
    history["SALES_GROWTH_EVENT_TIME"] = _latest_time(
        history,
        ["REVENUE_EVENT_TIME", "LAG4_REVENUE_EVENT_TIME"],
    )

    pieces: list[pd.DataFrame] = []
    for _, group in history.groupby("SECURITY_ID", sort=False):
        indexed = group.set_index("QUARTER_INDEX").sort_index()
        full_index = pd.RangeIndex(
            int(indexed.index.min()),
            int(indexed.index.max()) + 1,
        )
        full = indexed.reindex(full_index)
        piece = pd.DataFrame(
            {
                "SECURITY_ID": int(group["SECURITY_ID"].iloc[0]),
                "QUARTER_INDEX": full_index,
            }
        )
        for value_column, event_column, output_column in [
            (
                "QUARTERLY_ROA",
                "QUARTERLY_ROA_EVENT_TIME",
                "VAR_ROA",
            ),
            (
                "SALES_GROWTH",
                "SALES_GROWTH_EVENT_TIME",
                "VAR_SALES_GROWTH",
            ),
        ]:
            values = pd.to_numeric(full[value_column], errors="coerce")
            piece[output_column] = (
                values.rolling(
                    HISTORY_QUARTERS,
                    min_periods=MIN_HISTORY_OBSERVATIONS,
                )
                .var(ddof=1)
                .to_numpy()
            )
            times = pd.to_datetime(full[event_column], errors="coerce")
            time_ns = pd.Series(
                times.astype("int64").to_numpy(dtype=np.float64),
                index=full_index,
            ).where(times.notna())
            available_ns = time_ns.rolling(
                HISTORY_QUARTERS,
                min_periods=MIN_HISTORY_OBSERVATIONS,
            ).max()
            piece[f"{output_column}_EVENT_TIME"] = pd.to_datetime(
                available_ns, unit="ns", errors="coerce"
            ).to_numpy()
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def build_raw_g_metrics(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    balance: pd.DataFrame,
) -> pd.DataFrame:
    """Construct firm-quarter raw inputs before contextual peer scoring."""
    flows = build_quarterly_flows(income, cashflow)
    balances = build_quarterly_balance(balance)
    ttm = _build_ttm_flows(flows)
    history = _build_quarterly_history(flows, balances)

    keys = [
        "SECURITY_ID",
        "FISCAL_YEAR",
        "FISCAL_QUARTER",
        "QUARTER_INDEX",
    ]
    current = balances[
        keys
        + [
            "END_DATE",
            "T_ASSETS",
            "BALANCE_EVENT_TIME",
            "INDUSTRY_CATEGORY",
        ]
    ]
    lag4 = balances[
        ["SECURITY_ID", "QUARTER_INDEX", "T_ASSETS", "BALANCE_EVENT_TIME"]
    ].copy()
    lag4["QUARTER_INDEX"] += 4
    lag4 = lag4.rename(
        columns={
            "T_ASSETS": "BEGINNING_ASSETS",
            "BALANCE_EVENT_TIME": "BEGINNING_ASSETS_EVENT_TIME",
        }
    )
    metrics = pd.merge(
        ttm,
        current,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    metrics = pd.merge(
        metrics,
        lag4,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    metrics = pd.merge(
        metrics,
        history,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="left",
        validate="one_to_one",
    )

    average_assets = (metrics["T_ASSETS"] + metrics["BEGINNING_ASSETS"]) / 2
    valid_average = average_assets.gt(0)
    valid_beginning = metrics["BEGINNING_ASSETS"].gt(0)
    metrics["ROA"] = (
        metrics["TTM_N_INCOME_ATTR_P"].div(average_assets)
        .where(valid_average)
    )
    metrics["CFROA"] = (
        metrics["TTM_N_CF_OPERATE_A"].div(average_assets)
        .where(valid_average)
    )
    metrics["RD_INTENSITY"] = (
        metrics["TTM_R_D_EXP"].clip(lower=0)
        .div(metrics["BEGINNING_ASSETS"])
        .where(valid_beginning)
    )
    metrics["CAPEX_INTENSITY"] = (
        metrics["TTM_PUR_FIX_ASSETS_OTH"].clip(lower=0)
        .div(metrics["BEGINNING_ASSETS"])
        .where(valid_beginning)
    )
    metrics["AD_PROXY_INTENSITY"] = (
        metrics["TTM_SELL_EXP"].clip(lower=0)
        .div(metrics["BEGINNING_ASSETS"])
        .where(valid_beginning)
    )
    metrics["METRIC_EVENT_TIME"] = _latest_time(
        metrics,
        [
            "TTM_EVENT_TIME",
            "BALANCE_EVENT_TIME",
            "BEGINNING_ASSETS_EVENT_TIME",
            "VAR_ROA_EVENT_TIME",
            "VAR_SALES_GROWTH_EVENT_TIME",
        ],
    )
    required = [
        "ROA",
        "CFROA",
        "RD_INTENSITY",
        "CAPEX_INTENSITY",
        "AD_PROXY_INTENSITY",
        "METRIC_EVENT_TIME",
    ]
    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna(
        subset=required
    )
    return metrics.sort_values(
        ["SECURITY_ID", "QUARTER_INDEX"]
    ).reset_index(drop=True)


def _barra_industry_columns(barra_path: Path) -> list[str]:
    names = pq.ParquetFile(barra_path).schema_arrow.names
    excluded = set(KEYS + STYLE_COLUMNS)
    return [name for name in names if name not in excluded]


def _read_barra_year_context(
    barra_path: Path,
    year: int,
    event_dates: pd.DatetimeIndex,
    industry_columns: list[str],
) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    data = pd.read_parquet(
        barra_path,
        columns=KEYS + ["book_to_price"] + industry_columns,
        filters=[
            ("TRADE_DATE", ">=", start),
            ("TRADE_DATE", "<=", end),
        ],
        engine="pyarrow",
    )
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"]).dt.normalize()
    data = data.loc[data["TRADE_DATE"].isin(event_dates)].copy()
    if data.empty:
        return data
    matrix = data[industry_columns].to_numpy(dtype=np.float64)
    usable = np.isfinite(matrix)
    safe = np.where(usable, matrix, -np.inf)
    positions = safe.argmax(axis=1)
    maximum = safe[np.arange(len(safe)), positions]
    labels = np.asarray(industry_columns, dtype=object)[positions]
    labels[~np.isfinite(maximum)] = "未分类"
    data["BARRA_INDUSTRY"] = labels
    return data[KEYS + ["book_to_price", "BARRA_INDUSTRY"]]


def _score_event_rows(
    updates: pd.DataFrame,
    context: pd.DataFrame,
    state: pd.DataFrame,
    growth_cutoff: float,
) -> pd.DataFrame:
    """Score today's report events against today's PIT growth peers."""
    peer = state.join(
        context[["book_to_price", "BARRA_INDUSTRY"]],
        how="inner",
    )
    peer = peer.loc[
        pd.to_numeric(peer["book_to_price"], errors="coerce").le(
            growth_cutoff
        )
        & peer["BARRA_INDUSTRY"].ne("未分类")
    ].copy()
    grouped = peer.groupby("BARRA_INDUSTRY", sort=False)
    medians = grouped[PEER_METRICS].median().add_prefix("MEDIAN_")
    counts = grouped[PEER_METRICS].count().add_prefix("COUNT_")
    benchmarks = medians.join(counts)

    scored = updates.set_index("SECURITY_ID", drop=False).join(
        context[["book_to_price", "BARRA_INDUSTRY"]],
        how="left",
    )
    scored = scored.join(benchmarks, on="BARRA_INDUSTRY")
    scored["GROWTH_CUTOFF"] = growth_cutoff
    scored["IS_GROWTH"] = (
        scored["book_to_price"].le(growth_cutoff)
        & scored["BARRA_INDUSTRY"].ne("未分类")
    )

    core_peer_metrics = [
        "ROA",
        "CFROA",
        "RD_INTENSITY",
        "CAPEX_INTENSITY",
        "AD_PROXY_INTENSITY",
    ]
    peer_valid = pd.Series(True, index=scored.index)
    for metric in core_peer_metrics:
        peer_valid &= scored[f"COUNT_{metric}"].ge(MIN_INDUSTRY_FIRMS)
        peer_valid &= scored[f"MEDIAN_{metric}"].notna()
    scored["PEER_VALID"] = peer_valid
    eligible = scored["IS_GROWTH"] & scored["PEER_VALID"]

    scored["G1_ROA"] = (
        scored["ROA"].ge(scored["MEDIAN_ROA"]) & eligible
    )
    scored["G2_CFROA"] = (
        scored["CFROA"].ge(scored["MEDIAN_CFROA"]) & eligible
    )
    scored["G3_CFO_ABOVE_INCOME"] = (
        scored["TTM_N_CF_OPERATE_A"].gt(
            scored["TTM_N_INCOME_ATTR_P"]
        )
        & eligible
    )
    scored["G4_EARNINGS_STABILITY"] = (
        scored["VAR_ROA"].notna()
        & scored["COUNT_VAR_ROA"].ge(MIN_INDUSTRY_FIRMS)
        & scored["VAR_ROA"].le(scored["MEDIAN_VAR_ROA"])
        & eligible
    )
    scored["G5_GROWTH_STABILITY"] = (
        scored["VAR_SALES_GROWTH"].notna()
        & scored["COUNT_VAR_SALES_GROWTH"].ge(MIN_INDUSTRY_FIRMS)
        & scored["VAR_SALES_GROWTH"].le(
            scored["MEDIAN_VAR_SALES_GROWTH"]
        )
        & eligible
    )
    scored["G6_RD_INTENSITY"] = (
        scored["RD_INTENSITY"].ge(scored["MEDIAN_RD_INTENSITY"])
        & eligible
    )
    scored["G7_CAPEX_INTENSITY"] = (
        scored["CAPEX_INTENSITY"].ge(
            scored["MEDIAN_CAPEX_INTENSITY"]
        )
        & eligible
    )
    scored["G8_AD_PROXY_INTENSITY"] = (
        scored["AD_PROXY_INTENSITY"].ge(
            scored["MEDIAN_AD_PROXY_INTENSITY"]
        )
        & eligible
    )
    for column in SIGNAL_COLUMNS:
        scored[column] = scored[column].astype("int8")
    scored["MOHANRAM_G_SCORE"] = (
        scored[SIGNAL_COLUMNS].sum(axis=1).astype("float64")
    ).where(eligible)
    return scored.reset_index(drop=True)


def build_contextual_g_score_events(
    raw_metrics: pd.DataFrame,
    factor_calendar: pd.Series,
    barra_path: str | Path,
) -> pd.DataFrame:
    """Apply daily PIT growth-universe and industry-median comparisons."""
    path = Path(barra_path).resolve()
    available = assign_available_trade_date(
        raw_metrics.rename(columns={"METRIC_EVENT_TIME": "EVENT_TIME"}),
        factor_calendar,
    )
    available = available.sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
    )
    latest_quarter = available.groupby("SECURITY_ID")[
        "QUARTER_INDEX"
    ].cummax()
    available = available.loc[
        available["QUARTER_INDEX"].eq(latest_quarter)
    ]
    available = available.drop_duplicates(
        ["SECURITY_ID", "AVAILABLE_DATE"], keep="last"
    ).sort_values(["AVAILABLE_DATE", "EVENT_TIME", "SECURITY_ID"])

    industry_columns = _barra_industry_columns(path)
    state = pd.DataFrame().rename_axis("SECURITY_ID")
    scored_pieces: list[pd.DataFrame] = []
    for year in sorted(available["AVAILABLE_DATE"].dt.year.unique()):
        year_events = available.loc[
            available["AVAILABLE_DATE"].dt.year.eq(year)
        ]
        dates = pd.DatetimeIndex(year_events["AVAILABLE_DATE"].unique())
        context_year = _read_barra_year_context(
            path,
            int(year),
            dates,
            industry_columns,
        )
        context_groups = {
            date: group.drop(columns="TRADE_DATE")
            .drop_duplicates("SECURITY_ID", keep="last")
            .set_index("SECURITY_ID")
            for date, group in context_year.groupby(
                "TRADE_DATE", sort=False
            )
        }
        for date, updates in year_events.groupby(
            "AVAILABLE_DATE", sort=True
        ):
            indexed_updates = updates.set_index("SECURITY_ID", drop=False)
            state = pd.concat(
                [
                    state.drop(index=indexed_updates.index, errors="ignore"),
                    indexed_updates,
                ],
                axis=0,
            )
            context = context_groups.get(pd.Timestamp(date))
            if context is None or context.empty:
                empty = updates.copy()
                empty["MOHANRAM_G_SCORE"] = np.nan
                empty["IS_GROWTH"] = False
                empty["PEER_VALID"] = False
                scored_pieces.append(empty)
                continue
            book_to_price = pd.to_numeric(
                context["book_to_price"], errors="coerce"
            ).dropna()
            if len(book_to_price) < 20:
                empty = updates.copy()
                empty["MOHANRAM_G_SCORE"] = np.nan
                empty["IS_GROWTH"] = False
                empty["PEER_VALID"] = False
                scored_pieces.append(empty)
                continue
            cutoff = float(book_to_price.quantile(GROWTH_QUANTILE))
            scored_pieces.append(
                _score_event_rows(updates, context, state, cutoff)
            )
        del context_year, context_groups
        gc.collect()

    result = pd.concat(scored_pieces, ignore_index=True)
    return result.sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "QUARTER_INDEX"]
    ).reset_index(drop=True)


def build_daily_g_score(
    scored_events: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    daily = panel[KEYS].copy()
    daily["TRADE_DATE"] = pd.to_datetime(
        daily["TRADE_DATE"], errors="coerce"
    ).dt.normalize()
    daily["SECURITY_ID"] = pd.to_numeric(
        daily["SECURITY_ID"], errors="coerce"
    )
    daily = daily.dropna(subset=KEYS)
    daily["SECURITY_ID"] = daily["SECURITY_ID"].astype("int64")
    if daily.duplicated(KEYS).any():
        raise ValueError("factors.parquet存在重复证券-交易日键")

    event_groups = {
        int(security_id): group.sort_values(
            ["AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
        ).drop_duplicates("AVAILABLE_DATE", keep="last")
        for security_id, group in scored_events.groupby(
            "SECURITY_ID", sort=False
        )
    }
    pieces: list[pd.DataFrame] = []
    for security_id, stock_days in daily.groupby(
        "SECURITY_ID", sort=False
    ):
        left = stock_days.sort_values("TRADE_DATE")
        right = event_groups.get(int(security_id))
        if right is None or right.empty:
            pieces.append(left.assign(**{FACTOR_NAME: np.nan}))
            continue
        joined = pd.merge_asof(
            left,
            right[["AVAILABLE_DATE", "MOHANRAM_G_SCORE"]],
            left_on="TRADE_DATE",
            right_on="AVAILABLE_DATE",
            direction="backward",
        ).drop(columns="AVAILABLE_DATE")
        pieces.append(
            joined.rename(columns={"MOHANRAM_G_SCORE": FACTOR_NAME})
        )
    return pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(
        drop=True
    )


def append_factor_atomically(
    factor_path: str | Path,
    factor_values: pd.DataFrame,
) -> tuple[Path, Path]:
    path = Path(factor_path).resolve()
    existing = pd.read_parquet(path)
    existing["TRADE_DATE"] = pd.to_datetime(
        existing["TRADE_DATE"], errors="coerce"
    ).dt.normalize()
    existing["SECURITY_ID"] = pd.to_numeric(
        existing["SECURITY_ID"], errors="raise"
    ).astype("int64")
    if existing.duplicated(KEYS).any():
        raise ValueError("原factors.parquet存在重复键")
    values = factor_values[KEYS + [FACTOR_NAME]].copy()
    if values.duplicated(KEYS).any():
        raise ValueError("Mohanram G-score结果存在重复键")
    updated = pd.merge(
        existing.drop(columns=FACTOR_NAME, errors="ignore"),
        values,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(updated) != len(existing):
        raise RuntimeError("加入G-score后factors.parquet行数发生变化")
    backup_dir = path.parent / "输出与测试" / "因子备份"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (
        f"{path.stem}_before_mohanram_g_score{path.suffix}"
    )
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    updated.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    if pq.ParquetFile(temporary).metadata.num_rows != len(existing):
        raise RuntimeError("临时factors.parquet行数校验失败")
    os.replace(temporary, path)
    return path, backup


def run_mohanram_g_score(
    pit_dir: str | Path = DEFAULT_PIT_DIR,
    factor_path: str | Path = DEFAULT_FACTOR_PATH,
    barra_path: str | Path = DEFAULT_BARRA_PATH,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict:
    pit_dir = Path(pit_dir).resolve()
    factor_path = Path(factor_path).resolve()
    barra_path = Path(barra_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("读取三张PIT报表...")
    income, cashflow, balance = _read_and_prepare_statements(pit_dir)
    print(
        f"  income={len(income):,}, cashflow={len(cashflow):,}, "
        f"balance={len(balance):,}"
    )
    raw_metrics = build_raw_g_metrics(income, cashflow, balance)
    print(
        f"  raw firm-quarter metrics={len(raw_metrics):,}, "
        f"stocks={raw_metrics['SECURITY_ID'].nunique():,}"
    )
    del income, cashflow, balance
    gc.collect()

    factor_keys = pd.read_parquet(
        factor_path, columns=KEYS, engine="pyarrow"
    )
    scored_events = build_contextual_g_score_events(
        raw_metrics,
        factor_keys["TRADE_DATE"].unique(),
        barra_path,
    )
    valid_events = scored_events["MOHANRAM_G_SCORE"].notna()
    print(
        f"  contextual events={len(scored_events):,}, "
        f"valid growth-stock scores={int(valid_events.sum()):,}"
    )
    daily = build_daily_g_score(scored_events, factor_keys)
    output_path, backup_path = append_factor_atomically(factor_path, daily)

    raw_path = audit_dir / "mohanram_g_score_raw_metrics.parquet"
    event_path = audit_dir / "mohanram_g_score_events.parquet"
    daily_path = audit_dir / "mohanram_g_score_daily.parquet"
    diagnostics_path = audit_dir / "mohanram_g_score_diagnostics.json"
    raw_metrics.to_parquet(
        raw_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    scored_events.to_parquet(
        event_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    daily.to_parquet(
        daily_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    valid_daily = daily[FACTOR_NAME].notna()
    diagnostics = {
        "definition": (
            "quarterly PIT Mohanram G-score for the lowest daily "
            "book-to-price quintile"
        ),
        "g8_proxy": "SELL_EXP / beginning assets",
        "growth_quantile": GROWTH_QUANTILE,
        "minimum_industry_firms": MIN_INDUSTRY_FIRMS,
        "history_quarters": HISTORY_QUARTERS,
        "minimum_history_observations": MIN_HISTORY_OBSERVATIONS,
        "raw_metric_rows": len(raw_metrics),
        "contextual_event_rows": len(scored_events),
        "valid_event_rows": int(valid_events.sum()),
        "valid_event_stocks": int(
            scored_events.loc[valid_events, "SECURITY_ID"].nunique()
        ),
        "daily_panel_rows": len(daily),
        "daily_non_null": int(valid_daily.sum()),
        "daily_coverage": float(valid_daily.mean()),
        "factor_start": str(daily.loc[valid_daily, "TRADE_DATE"].min()),
        "factor_end": str(daily.loc[valid_daily, "TRADE_DATE"].max()),
        "score_distribution": {
            str(int(key)): int(value)
            for key, value in scored_events.loc[
                valid_events, "MOHANRAM_G_SCORE"
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "factor_path": str(output_path),
        "backup_path": str(backup_path),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return {
        "factor_path": output_path,
        "backup_path": backup_path,
        "raw_metrics_path": raw_path,
        "event_path": event_path,
        "daily_path": daily_path,
        "diagnostics_path": diagnostics_path,
        "diagnostics": diagnostics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造严格PIT的季度Mohanram G-score并加入factors.parquet"
    )
    parser.add_argument("--pit-dir", type=Path, default=DEFAULT_PIT_DIR)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--barra", type=Path, default=DEFAULT_BARRA_PATH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_mohanram_g_score(
        args.pit_dir,
        args.factors,
        args.barra,
        args.audit_dir,
    )


if __name__ == "__main__":
    main()
