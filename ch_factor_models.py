"""Reproduce the monthly CH-3 and CH-4 China factor models.

The implementation follows Liu, Stambaugh and Yuan, "Size and Value in
China":

* exclude the smallest 30% of eligible A shares before every factor sort;
* split the remaining universe at its size median;
* independently split EP and abnormal turnover 30/40/30;
* value-weight the six 2x3 portfolios;
* build MKT, SMB, VMG and PMO from next-month portfolio returns.

All characteristics and weights are measured at month end and are used only
for the following month's return.  Accounting and share-count observations
are assigned an availability date, preventing look-ahead bias.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import time, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "ch_models"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "ch_models"
DEFAULT_START_DATE = "2017-01-01"
DEFAULT_END_DATE = pd.Timestamp.today().normalize()


@dataclass(frozen=True)
class BuildConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    annual_risk_free: float = 0.015
    shell_cutoff: float = 0.30
    style_tail: float = 0.30
    min_list_days: int = 183
    min_records_year: int = 120
    min_records_month: int = 15
    min_portfolio_size: int = 50
    market_open: str = "09:30:00"

    @property
    def monthly_risk_free(self) -> float:
        return (1.0 + self.annual_risk_free) ** (1.0 / 12.0) - 1.0


KEYS = ["TRADE_DATE", "SECURITY_ID"]


def _dataset_files(root: Path, dataset_name: str) -> list[Path]:
    directory = root / dataset_name
    files = sorted(directory.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {directory}")
    return files


def read_parquet_dataset(
    files: Iterable[Path],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    paths = [str(path) for path in files]
    table = ds.dataset(paths, format="parquet").to_table(columns=columns)
    return table.to_pandas()


def assign_available_trade_date(
    events: pd.DataFrame,
    event_time_column: str,
    trading_calendar: pd.DatetimeIndex,
    market_open: str = "09:30:00",
) -> pd.DataFrame:
    """Map a timestamp to the first trading day on which it is observable."""
    result = events.copy()
    calendar = pd.DatetimeIndex(trading_calendar).normalize()
    calendar = calendar.drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("Trading calendar is empty")
    clock = time.fromisoformat(market_open)
    opening_delta = timedelta(
        hours=clock.hour,
        minutes=clock.minute,
        seconds=clock.second,
    )
    event_time = pd.to_datetime(
        result[event_time_column], errors="coerce"
    )
    event_day = event_time.dt.normalize()
    candidate = event_day.where(
        event_time.le(event_day + opening_delta),
        event_day + timedelta(days=1),
    )
    positions = calendar.searchsorted(candidate.to_numpy(), side="left")
    available = np.full(
        len(result),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    valid = positions < len(calendar)
    available[valid] = calendar.to_numpy()[positions[valid]]
    result["AVAILABLE_DATE"] = pd.to_datetime(available)
    return result.dropna(subset=["AVAILABLE_DATE"])


def normalize_market(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "TRADE_DATE",
        "SECURITY_ID",
        "LIST_DATE",
        "CLOSE_PRICE",
        "ADJ_CLOSE_PRICE",
        "TURNOVER_VOL",
        "MARKET_VALUE_A",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"market_daily is missing columns: {missing}")
    result = data.copy()
    for column in ("TRADE_DATE", "LIST_DATE", "DELIST_DATE"):
        if column in result:
            result[column] = pd.to_datetime(
                result[column], errors="coerce"
            )
    result["SECURITY_ID"] = pd.to_numeric(
        result["SECURITY_ID"], errors="raise"
    ).astype("int64")
    for column in (
        "CLOSE_PRICE",
        "ADJ_CLOSE_PRICE",
        "TURNOVER_VOL",
        "MARKET_VALUE_A",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(
        subset=[
            "TRADE_DATE",
            "SECURITY_ID",
            "CLOSE_PRICE",
            "ADJ_CLOSE_PRICE",
            "MARKET_VALUE_A",
        ]
    )
    result = result.loc[
        result["CLOSE_PRICE"].gt(0)
        & result["ADJ_CLOSE_PRICE"].gt(0)
        & result["MARKET_VALUE_A"].gt(0)
    ]
    result = result.drop_duplicates(KEYS, keep="last")
    return result.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(
        drop=True
    )


def add_trading_record_counts(market: pd.DataFrame) -> pd.DataFrame:
    """Count observed trading records in trailing calendar month/year."""
    result = market.sort_values(
        ["SECURITY_ID", "TRADE_DATE"]
    ).reset_index(drop=True).copy()
    records_31 = np.full(len(result), np.nan, dtype="float64")
    records_366 = np.full(len(result), np.nan, dtype="float64")
    for positions in result.groupby(
        "SECURITY_ID", sort=False
    ).indices.values():
        ordered = result.iloc[positions]
        indexed = ordered.set_index("TRADE_DATE")
        # Vendors can retain a daily quote row during suspension.  A positive
        # traded volume is therefore the appropriate definition of a trading
        # record for the paper's 15/120-record filters.
        marker = indexed["TURNOVER_VOL"].gt(0).astype("float64")
        records_31[positions] = marker.rolling("31D").sum().to_numpy()
        records_366[positions] = marker.rolling("366D").sum().to_numpy()
    result["RECORDS_31D"] = records_31
    result["RECORDS_366D"] = records_366
    return result


def add_abnormal_turnover(market: pd.DataFrame) -> pd.DataFrame:
    """Calculate turnover using total A shares, not free-float shares."""
    result = market.sort_values(
        ["SECURITY_ID", "TRADE_DATE"]
    ).reset_index(drop=True).copy()
    total_a_shares = result["MARKET_VALUE_A"].div(result["CLOSE_PRICE"])
    daily_turnover = result["TURNOVER_VOL"].div(total_a_shares)
    daily_turnover = daily_turnover.where(
        total_a_shares.gt(0) & result["TURNOVER_VOL"].ge(0)
    )
    result["DAILY_TURNOVER_TOTAL_A"] = daily_turnover
    average_20 = np.full(len(result), np.nan, dtype="float64")
    average_250 = np.full(len(result), np.nan, dtype="float64")
    for positions in result.groupby(
        "SECURITY_ID", sort=False
    ).indices.values():
        ordered = result.iloc[positions]
        series = ordered.set_index("TRADE_DATE")[
            "DAILY_TURNOVER_TOTAL_A"
        ]
        average_20[positions] = (
            series.rolling("31D", min_periods=15).mean().to_numpy()
        )
        average_250[positions] = (
            series.rolling("366D", min_periods=120).mean().to_numpy()
        )
    result["AVG_TURNOVER_1M"] = average_20
    result["AVG_TURNOVER_1Y"] = average_250
    result["ABNORMAL_TURNOVER"] = result["AVG_TURNOVER_1M"].div(
        result["AVG_TURNOVER_1Y"]
    )
    return result.replace([np.inf, -np.inf], np.nan)


def make_month_end_panel(market: pd.DataFrame) -> pd.DataFrame:
    result = market.copy()
    result["MONTH"] = result["TRADE_DATE"].dt.to_period("M")
    month_ends = (
        result.groupby("MONTH")["TRADE_DATE"].max().rename("MONTH_END_DATE")
    )
    result["MONTH_END_DATE"] = result["MONTH"].map(month_ends)
    panel = result.loc[
        result["TRADE_DATE"].eq(result["MONTH_END_DATE"])
    ].copy()

    adjusted = panel.sort_values(
        ["SECURITY_ID", "MONTH"]
    ).set_index(["SECURITY_ID", "MONTH"])["ADJ_CLOSE_PRICE"]
    returns = adjusted.groupby(level=0).pct_change(fill_method=None)
    panel = panel.set_index(["SECURITY_ID", "MONTH"])
    panel["MONTH_RETURN"] = returns
    panel = panel.reset_index()
    panel["LIST_DAYS"] = (
        panel["MONTH_END_DATE"] - panel["LIST_DATE"]
    ).dt.days
    return panel.sort_values(
        ["MONTH", "SECURITY_ID"]
    ).reset_index(drop=True)


def prepare_earnings(
    earnings: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    market_open: str,
) -> pd.DataFrame:
    required = {
        "SECURITY_ID",
        "ACT_PUBTIME",
        "PUBLISH_DATE",
        "END_DATE",
        "N_INCOME_CUT",
    }
    missing = sorted(required.difference(earnings.columns))
    if missing:
        raise KeyError(f"earnings_pit is missing columns: {missing}")
    result = earnings.copy()
    for column in ("ACT_PUBTIME", "PUBLISH_DATE", "END_DATE"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    # A date-only publication value has no reliable intraday timestamp.
    # Treat it as an after-close event and make it usable next trading day.
    date_only_fallback = (
        result["PUBLISH_DATE"].dt.normalize()
        + pd.to_timedelta("23:59:00")
    )
    result["EVENT_TIME"] = result["ACT_PUBTIME"].fillna(
        date_only_fallback
    )
    result["N_INCOME_CUT"] = pd.to_numeric(
        result["N_INCOME_CUT"], errors="coerce"
    )
    result = result.dropna(
        subset=["SECURITY_ID", "EVENT_TIME", "END_DATE", "N_INCOME_CUT"]
    )
    result["SECURITY_ID"] = result["SECURITY_ID"].astype("int64")
    result = result.sort_values(
        ["SECURITY_ID", "EVENT_TIME", "END_DATE", "ID"]
    ).drop_duplicates(
        ["SECURITY_ID", "EVENT_TIME", "END_DATE"], keep="last"
    )
    result = assign_available_trade_date(
        result,
        "EVENT_TIME",
        trading_calendar,
        market_open=market_open,
    )
    return result.sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "END_DATE"]
    )


def prepare_shares(
    shares: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    market_open: str,
) -> pd.DataFrame:
    required = {
        "SECURITY_ID",
        "PUBLISH_DATE",
        "CHANGE_DATE",
        "TOTAL_SHARES",
    }
    missing = sorted(required.difference(shares.columns))
    if missing:
        raise KeyError(f"share_changes is missing columns: {missing}")
    result = shares.copy()
    for column in ("PUBLISH_DATE", "CHANGE_DATE"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["TOTAL_SHARES"] = pd.to_numeric(
        result["TOTAL_SHARES"], errors="coerce"
    )
    result["SECURITY_ID"] = pd.to_numeric(
        result["SECURITY_ID"], errors="raise"
    ).astype("int64")
    # A share count becomes usable only after both its effective date and
    # publication date have arrived.
    effective_day = result[
        ["PUBLISH_DATE", "CHANGE_DATE"]
    ].max(axis=1)
    # Both share dates are date-only fields.  A next-trading-day convention
    # is conservative and prevents same-day look-ahead.
    result["EVENT_TIME"] = (
        effective_day.dt.normalize()
        + pd.to_timedelta("23:59:00")
    )
    result = result.dropna(
        subset=["EVENT_TIME", "TOTAL_SHARES"]
    ).loc[result["TOTAL_SHARES"].gt(0)]
    result = result.sort_values(
        ["SECURITY_ID", "EVENT_TIME", "ID"]
    ).drop_duplicates(
        ["SECURITY_ID", "EVENT_TIME"], keep="last"
    )
    result = assign_available_trade_date(
        result,
        "EVENT_TIME",
        trading_calendar,
        market_open=market_open,
    )
    return result.sort_values(["SECURITY_ID", "AVAILABLE_DATE", "ID"])


def merge_latest_by_security(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    event_columns: list[str],
) -> pd.DataFrame:
    left = panel.sort_values(["SECURITY_ID", "MONTH_END_DATE"]).copy()
    right_columns = ["SECURITY_ID", "AVAILABLE_DATE", *event_columns]
    right = events[right_columns].sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE"]
    )
    pieces: list[pd.DataFrame] = []
    right_groups = {
        int(key): value.drop(columns="SECURITY_ID").sort_values(
            "AVAILABLE_DATE"
        )
        for key, value in right.groupby("SECURITY_ID", sort=False)
    }
    for security_id, group in left.groupby("SECURITY_ID", sort=False):
        event_group = right_groups.get(int(security_id))
        if event_group is None or event_group.empty:
            output = group.copy()
            for column in ["AVAILABLE_DATE", *event_columns]:
                output[column] = pd.NaT if "DATE" in column else np.nan
        else:
            output = pd.merge_asof(
                group.sort_values("MONTH_END_DATE"),
                event_group,
                left_on="MONTH_END_DATE",
                right_on="AVAILABLE_DATE",
                direction="backward",
                allow_exact_matches=True,
            )
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True)


def add_ep(
    panel: pd.DataFrame,
    earnings: pd.DataFrame,
    shares: pd.DataFrame,
) -> pd.DataFrame:
    result = merge_latest_by_security(
        panel,
        earnings,
        ["N_INCOME_CUT", "END_DATE"],
    ).rename(
        columns={
            "AVAILABLE_DATE": "EARNINGS_AVAILABLE_DATE",
            "END_DATE": "EARNINGS_END_DATE",
        }
    )
    result = merge_latest_by_security(
        result,
        shares,
        ["TOTAL_SHARES"],
    ).rename(columns={"AVAILABLE_DATE": "SHARES_AVAILABLE_DATE"})

    inferred_a_shares = result["MARKET_VALUE_A"].div(
        result["CLOSE_PRICE"]
    )
    result["EP_TOTAL_SHARES"] = result["TOTAL_SHARES"].where(
        result["TOTAL_SHARES"].gt(0), inferred_a_shares
    )
    result["EP_SHARE_SOURCE"] = np.where(
        result["TOTAL_SHARES"].gt(0),
        "reported_total_shares",
        "inferred_a_shares_fallback",
    )
    denominator = result["CLOSE_PRICE"].mul(result["EP_TOTAL_SHARES"])
    result["EP"] = result["N_INCOME_CUT"].div(denominator).where(
        denominator.gt(0)
    )
    return result.replace([np.inf, -np.inf], np.nan)


def weighted_return(
    group: pd.DataFrame,
    return_column: str = "MONTH_RETURN",
    weight_column: str = "FORMATION_MARKET_VALUE_A",
) -> float:
    valid = group[[return_column, weight_column]].dropna()
    valid = valid.loc[valid[weight_column].gt(0)]
    if valid.empty:
        return np.nan
    return float(
        np.average(valid[return_column], weights=valid[weight_column])
    )


def _style_bucket(
    values: pd.Series,
    low_name: str,
    middle_name: str,
    high_name: str,
    tail: float,
    force_negative_low: bool = False,
) -> pd.Series:
    output = pd.Series(pd.NA, index=values.index, dtype="string")
    valid = values.dropna()
    if valid.empty:
        return output
    lower = float(valid.quantile(tail))
    upper = float(valid.quantile(1.0 - tail))
    output.loc[valid.index[valid.le(lower)]] = low_name
    output.loc[valid.index[valid.ge(upper)]] = high_name
    middle = valid.index[valid.gt(lower) & valid.lt(upper)]
    output.loc[middle] = middle_name
    if force_negative_low:
        output.loc[valid.index[valid.lt(0)]] = low_name
    return output


def form_portfolios(
    formation: pd.DataFrame,
    config: BuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign the six EP and six turnover portfolios at each month end."""
    assigned: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for formation_month, group in formation.groupby("MONTH", sort=True):
        current = group.copy()
        raw_count = len(current)
        eligible = (
            current["LIST_DAYS"].ge(config.min_list_days)
            & current["RECORDS_366D"].ge(config.min_records_year)
            & current["RECORDS_31D"].ge(config.min_records_month)
            & current["MARKET_VALUE_A"].gt(0)
        )
        current = current.loc[eligible].copy()
        eligible_count = len(current)
        if current.empty:
            continue

        cutoff = float(
            current["MARKET_VALUE_A"].quantile(config.shell_cutoff)
        )
        current["SHELL_EXCLUDED"] = current["MARKET_VALUE_A"].le(cutoff)
        current = current.loc[~current["SHELL_EXCLUDED"]].copy()
        after_shell_count = len(current)
        if current.empty:
            continue

        size_median = float(current["MARKET_VALUE_A"].median())
        current["SIZE_GROUP"] = np.where(
            current["MARKET_VALUE_A"].le(size_median), "S", "B"
        )
        current["EP_GROUP"] = _style_bucket(
            current["EP"],
            low_name="G",
            middle_name="M",
            high_name="V",
            tail=config.style_tail,
            force_negative_low=True,
        )
        current["TURNOVER_GROUP"] = _style_bucket(
            current["ABNORMAL_TURNOVER"],
            low_name="P",
            middle_name="N",
            high_name="O",
            tail=config.style_tail,
        )
        current["EP_PORTFOLIO"] = (
            current["SIZE_GROUP"] + current["EP_GROUP"].fillna("")
        ).where(current["EP_GROUP"].notna())
        current["TURNOVER_PORTFOLIO"] = (
            current["SIZE_GROUP"] + current["TURNOVER_GROUP"].fillna("")
        ).where(current["TURNOVER_GROUP"].notna())

        counts_ep = current["EP_PORTFOLIO"].value_counts()
        counts_turnover = current["TURNOVER_PORTFOLIO"].value_counts()
        diagnostics.append(
            {
                "FORMATION_MONTH": str(formation_month),
                "FORMATION_DATE": current["MONTH_END_DATE"].max(),
                "raw_month_end_rows": raw_count,
                "eligible_rows": eligible_count,
                "after_shell_cutoff_rows": after_shell_count,
                "shell_market_cap_cutoff": cutoff,
                "size_median": size_median,
                "ep_valid_rows": int(current["EP"].notna().sum()),
                "turnover_valid_rows": int(
                    current["ABNORMAL_TURNOVER"].notna().sum()
                ),
                "negative_ep_rows": int(current["EP"].lt(0).sum()),
                "reported_total_shares_rows": int(
                    current["EP_SHARE_SOURCE"]
                    .eq("reported_total_shares")
                    .sum()
                ),
                **{
                    f"n_ep_{name}": int(counts_ep.get(name, 0))
                    for name in ("SV", "SM", "SG", "BV", "BM", "BG")
                },
                **{
                    f"n_turnover_{name}": int(
                        counts_turnover.get(name, 0)
                    )
                    for name in ("SP", "SN", "SO", "BP", "BN", "BO")
                },
            }
        )
        assigned.append(current)

    if not assigned:
        raise ValueError("No valid formation portfolios were produced")
    return (
        pd.concat(assigned, ignore_index=True),
        pd.DataFrame(diagnostics),
    )


def _portfolio_return_table(
    assignments: pd.DataFrame,
    portfolio_column: str,
    names: tuple[str, ...],
    prefix: str,
    min_portfolio_size: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for holding_date, date_group in assignments.groupby(
        "RETURN_MONTH_END_DATE", sort=True
    ):
        row: dict = {"TRADE_DATE": holding_date}
        for name in names:
            portfolio = date_group.loc[
                date_group[portfolio_column].eq(name)
            ]
            row[f"{prefix}_{name}_N"] = len(portfolio)
            row[f"{prefix}_{name}"] = (
                weighted_return(portfolio)
                if len(portfolio) >= min_portfolio_size
                else np.nan
            )
        records.append(row)
    return pd.DataFrame(records).sort_values("TRADE_DATE")


def build_factor_returns(
    assignments: pd.DataFrame,
    config: BuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ep = _portfolio_return_table(
        assignments,
        "EP_PORTFOLIO",
        ("SV", "SM", "SG", "BV", "BM", "BG"),
        "EP",
        config.min_portfolio_size,
    )
    turnover = _portfolio_return_table(
        assignments,
        "TURNOVER_PORTFOLIO",
        ("SP", "SN", "SO", "BP", "BN", "BO"),
        "TO",
        config.min_portfolio_size,
    )
    portfolios = ep.merge(turnover, on="TRADE_DATE", how="outer")

    market = (
        assignments.groupby("RETURN_MONTH_END_DATE", sort=True)
        .apply(weighted_return, include_groups=False)
        .rename("MARKET_RETURN")
        .reset_index()
        .rename(columns={"RETURN_MONTH_END_DATE": "TRADE_DATE"})
    )
    factors = portfolios.merge(market, on="TRADE_DATE", how="left")
    factors["RF"] = config.monthly_risk_free
    factors["MKT"] = factors["MARKET_RETURN"] - factors["RF"]
    factors["SMB_EP"] = factors[
        ["EP_SV", "EP_SM", "EP_SG"]
    ].mean(axis=1, skipna=False) - factors[
        ["EP_BV", "EP_BM", "EP_BG"]
    ].mean(
        axis=1, skipna=False
    )
    factors["VMG"] = factors[["EP_SV", "EP_BV"]].mean(
        axis=1, skipna=False
    ) - factors[["EP_SG", "EP_BG"]].mean(axis=1, skipna=False)
    factors["SMB_TURNOVER"] = factors[
        ["TO_SP", "TO_SN", "TO_SO"]
    ].mean(axis=1, skipna=False) - factors[
        ["TO_BP", "TO_BN", "TO_BO"]
    ].mean(
        axis=1, skipna=False
    )
    factors["PMO"] = factors[["TO_SP", "TO_BP"]].mean(
        axis=1, skipna=False
    ) - factors[["TO_SO", "TO_BO"]].mean(axis=1, skipna=False)
    factors["SMB_CH4"] = factors[["SMB_EP", "SMB_TURNOVER"]].mean(
        axis=1, skipna=False
    )

    ch3 = factors[["TRADE_DATE", "MKT", "SMB_EP", "VMG"]].rename(
        columns={"SMB_EP": "SMB"}
    )
    ch4 = factors[
        ["TRADE_DATE", "MKT", "SMB_CH4", "VMG", "PMO"]
    ].rename(columns={"SMB_CH4": "SMB"})
    return ch3, ch4, factors


def factor_summary(factors: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in factors.columns
        if column != "TRADE_DATE"
        and pd.api.types.is_numeric_dtype(factors[column])
    ]
    rows: list[dict] = []
    for column in columns:
        values = factors[column].dropna()
        count = len(values)
        mean = float(values.mean()) if count else np.nan
        std = float(values.std(ddof=1)) if count > 1 else np.nan
        t_stat = (
            mean / (std / math.sqrt(count))
            if count > 1 and std and np.isfinite(std)
            else np.nan
        )
        rows.append(
            {
                "factor": column,
                "months": count,
                "mean_monthly": mean,
                "std_monthly": std,
                "t_stat_mean": t_stat,
                "annualized_mean_simple": 12.0 * mean
                if np.isfinite(mean)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_models(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    annual_risk_free: float = 0.015,
    min_portfolio_size: int = 50,
) -> dict[str, Path]:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = BuildConfig(
        start_date=pd.Timestamp(start_date).normalize(),
        end_date=pd.Timestamp(end_date).normalize(),
        annual_risk_free=float(annual_risk_free),
        min_portfolio_size=int(min_portfolio_size),
    )
    # A factor observation represents a complete calendar month.  If the
    # requested end date is inside an unfinished month, stop at the previous
    # month end instead of publishing a partial-month return.
    factor_end_date = (
        config.end_date
        if config.end_date.is_month_end
        else config.end_date - pd.offsets.MonthEnd(1)
    )

    print("Loading local Parquet inputs")
    market = normalize_market(
        read_parquet_dataset(
            _dataset_files(input_root, "market_daily"),
            columns=[
                "TRADE_DATE",
                "SECURITY_ID",
                "TICKER_SYMBOL",
                "EXCHANGE_CD",
                "LIST_DATE",
                "DELIST_DATE",
                "CLOSE_PRICE",
                "ADJ_CLOSE_PRICE",
                "TURNOVER_VOL",
                "MARKET_VALUE_A",
            ],
        )
    )
    warmup_start = config.start_date - timedelta(days=400)
    market = market.loc[
        market["TRADE_DATE"].between(warmup_start, config.end_date)
    ].copy()
    earnings = read_parquet_dataset(
        _dataset_files(input_root, "earnings_pit")
    )
    shares = read_parquet_dataset(
        _dataset_files(input_root, "share_changes")
    )

    print("Computing trading-history and abnormal-turnover measures")
    market = add_trading_record_counts(market)
    market = add_abnormal_turnover(market)
    panel = make_month_end_panel(market)
    calendar = pd.DatetimeIndex(market["TRADE_DATE"].drop_duplicates())

    print("Applying PIT earnings and total-share histories")
    earnings = prepare_earnings(
        earnings, calendar, market_open=config.market_open
    )
    shares = prepare_shares(
        shares, calendar, market_open=config.market_open
    )
    panel = add_ep(panel, earnings, shares)

    # Characteristics from month t are paired with returns in month t+1.
    panel["FORMATION_MARKET_VALUE_A"] = panel["MARKET_VALUE_A"]
    panel["MONTH_RETURN_NEXT"] = panel.groupby("SECURITY_ID")[
        "MONTH_RETURN"
    ].shift(-1)
    panel["RETURN_MONTH_END_DATE"] = panel.groupby("SECURITY_ID")[
        "MONTH_END_DATE"
    ].shift(-1)
    panel["RETURN_MONTH"] = panel.groupby("SECURITY_ID")["MONTH"].shift(-1)
    consecutive = panel["RETURN_MONTH"].eq(panel["MONTH"] + 1)
    panel["MONTH_RETURN_NEXT"] = panel["MONTH_RETURN_NEXT"].where(
        consecutive
    )
    panel["RETURN_MONTH_END_DATE"] = panel[
        "RETURN_MONTH_END_DATE"
    ].where(consecutive)
    formation_start = (
        config.start_date.to_period("M") - 1
    ).start_time
    panel = panel.loc[
        panel["MONTH_END_DATE"].between(formation_start, config.end_date)
    ].copy()

    print("Forming monthly 2x3 portfolios")
    assignments, diagnostics = form_portfolios(panel, config)
    assignments["MONTH_RETURN"] = assignments["MONTH_RETURN_NEXT"]
    ch3, ch4, portfolios = build_factor_returns(assignments, config)
    ch3 = ch3.loc[
        ch3["TRADE_DATE"].between(config.start_date, factor_end_date)
    ].reset_index(drop=True)
    ch4 = ch4.loc[
        ch4["TRADE_DATE"].between(config.start_date, factor_end_date)
    ].reset_index(drop=True)
    portfolios = portfolios.loc[
        portfolios["TRADE_DATE"].between(
            config.start_date, factor_end_date
        )
    ].reset_index(drop=True)

    paths = {
        "ch3_parquet": output_root / "ch3_factors.parquet",
        "ch3_csv": output_root / "ch3_factors.csv",
        "ch4_parquet": output_root / "ch4_factors.parquet",
        "ch4_csv": output_root / "ch4_factors.csv",
        "portfolio_returns": output_root / "portfolio_returns.parquet",
        "assignments": output_root / "formation_assignments.parquet",
        "diagnostics": output_root / "monthly_diagnostics.parquet",
        "summary": output_root / "factor_summary.csv",
        "ch3_correlations": output_root / "ch3_factor_correlations.csv",
        "ch4_correlations": output_root / "ch4_factor_correlations.csv",
        "metadata": output_root / "run_metadata.json",
    }
    ch3.to_parquet(paths["ch3_parquet"], index=False)
    ch3.to_csv(paths["ch3_csv"], index=False)
    ch4.to_parquet(paths["ch4_parquet"], index=False)
    ch4.to_csv(paths["ch4_csv"], index=False)
    portfolios.to_parquet(paths["portfolio_returns"], index=False)
    assignments.to_parquet(paths["assignments"], index=False)
    diagnostics.to_parquet(paths["diagnostics"], index=False)
    ch3_summary = factor_summary(ch3)
    ch3_summary.insert(0, "model", "CH-3")
    ch4_summary = factor_summary(ch4)
    ch4_summary.insert(0, "model", "CH-4")
    pd.concat([ch3_summary, ch4_summary], ignore_index=True).to_csv(
        paths["summary"], index=False
    )
    ch3.drop(columns="TRADE_DATE").corr().to_csv(
        paths["ch3_correlations"]
    )
    ch4.drop(columns="TRADE_DATE").corr().to_csv(
        paths["ch4_correlations"]
    )

    metadata = {
        "paper": "Liu, Stambaugh and Yuan, Size and Value in China",
        "model_frequency": "monthly",
        "start_date": str(config.start_date.date()),
        "requested_end_date": str(config.end_date.date()),
        "last_complete_month_end": str(factor_end_date.date()),
        "annual_risk_free": config.annual_risk_free,
        "monthly_risk_free": config.monthly_risk_free,
        "shell_cutoff": config.shell_cutoff,
        "style_tail": config.style_tail,
        "min_list_days": config.min_list_days,
        "min_records_year": config.min_records_year,
        "min_records_month": config.min_records_month,
        "min_portfolio_size": config.min_portfolio_size,
        "ep_numerator": "PIT N_INCOME_CUT",
        "ep_denominator": (
            "month-end close times PIT total shares including A/B/H; "
            "A-share inferred fallback is disclosed in assignments"
        ),
        "turnover": (
            "average daily total-A-share turnover over 31 calendar days "
            "divided by the corresponding 366-day average"
        ),
        "formation_lag": "month-t characteristics, month-t+1 returns",
        "ch3_rows": len(ch3),
        "ch4_rows": len(ch4),
        "valid_ch3_rows": int(ch3.dropna().shape[0]),
        "valid_ch4_rows": int(ch4.dropna().shape[0]),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monthly CH-3 and CH-4 factor returns."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--end-date",
        default=str(pd.Timestamp(DEFAULT_END_DATE).date()),
    )
    parser.add_argument(
        "--annual-risk-free",
        type=float,
        default=0.015,
        help="One-year deposit rate in decimal form.",
    )
    parser.add_argument("--min-portfolio-size", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    outputs = build_models(
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        annual_risk_free=arguments.annual_risk_free,
        min_portfolio_size=arguments.min_portfolio_size,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
