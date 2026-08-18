"""Round 44: PIT total net payout yield, independent of profitability.

Net payout to capital providers is approximated from the Chinese cash-flow
statement as cash dividends/profit distributions/interest plus debt repayment,
less borrowing proceeds, bond issuance proceeds and cash capital contribution.
Quarterly and TTM versions are scaled by beginning assets.  A TTM year-on-year
improvement version uses only values already disclosed for t and t-4.

The first command emits natural missing values.  Filling is deliberately a
separate command and requires a sparse-test IC summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from event_financial_factor_search import KEYS, _normalize_panel
from ninth_round_independent_factors import (
    _flow_with_balance,
    _latest_time,
    _safe_ratio,
    build_balance_events,
)
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS, build_standalone_quarterly_metric


FACTORS = [
    "r44_q_total_net_payout_assets",
    "r44_ttm_total_net_payout_assets",
    "r44_ttm_total_net_payout_yoy_improvement",
]
FIELDS = [
    "C_PAID_DIV_PROF_INT", "C_PAID_FOR_DEBTS", "C_FR_BORR",
    "C_FR_ISSUE_BOND", "C_FR_CAP_CONTR",
]


def _payout(frame: pd.DataFrame, *, ttm: bool) -> pd.Series:
    def value(field: str) -> pd.Series:
        return frame[("TTM_" if ttm else "") + field]
    return (
        value("C_PAID_DIV_PROF_INT") + value("C_PAID_FOR_DEBTS")
        - value("C_FR_BORR") - value("C_FR_ISSUE_BOND")
        - value("C_FR_CAP_CONTR")
    )


def _event(frame: pd.DataFrame, factor: str, values: pd.Series, times: list[str]) -> pd.DataFrame:
    out = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    out["EVENT_TIME"] = _latest_time(frame, times)
    out["factor"] = factor
    out["value"] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["EVENT_TIME", "value"])


def calculate_events(pit_dir: Path) -> pd.DataFrame:
    cashflow = pd.read_parquet(
        pit_dir / "new_pit_cashflow", columns=COMMON_COLUMNS + FIELDS, engine="pyarrow"
    )
    # A disclosed cash-flow statement with an unreported optional financing
    # line means zero activity for that line, not an unavailable statement.
    cashflow[FIELDS] = cashflow[FIELDS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    flow_tables = {
        field: build_standalone_quarterly_metric(cashflow, field, name="cashflow PIT")
        for field in FIELDS
    }
    balance_fields = [
        "INDUSTRY_CATEGORY", "T_ASSETS", "CASH_C_EQUIV", "ST_BORR",
        "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE", "PAID_IN_CAPITAL",
        "T_EQUITY_ATTR_P", "RETAINED_EARNINGS", "INTAN_ASSETS", "GOODWILL",
    ]
    balance_raw = pd.read_parquet(
        pit_dir / "new_pit_balance", columns=COMMON_COLUMNS + balance_fields, engine="pyarrow"
    )
    balance = build_balance_events(balance_raw)
    q = _flow_with_balance(flow_tables, FIELDS, balance, ttm=False)
    q_value = _safe_ratio(_payout(q, ttm=False), q["L1_T_ASSETS"])
    q_event = _event(
        q, FACTORS[0], q_value, ["FLOW_EVENT_TIME", "L1_BALANCE_EVENT_TIME"]
    )

    ttm = _flow_with_balance(flow_tables, FIELDS, balance, ttm=True)
    ttm["payout_yield"] = _safe_ratio(_payout(ttm, ttm=True), ttm["L4_T_ASSETS"])
    ttm_event = _event(
        ttm, FACTORS[1], ttm["payout_yield"], ["FLOW_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]
    )
    prior = ttm[["SECURITY_ID", "QUARTER_INDEX", "payout_yield", "FLOW_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]].copy()
    prior["QUARTER_INDEX"] += 4
    prior = prior.rename(columns={
        "payout_yield": "prior_payout_yield",
        "FLOW_EVENT_TIME": "PRIOR_FLOW_EVENT_TIME",
        "L4_BALANCE_EVENT_TIME": "PRIOR_L4_BALANCE_EVENT_TIME",
    })
    change = ttm.merge(prior, on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    improvement = change["payout_yield"] - change["prior_payout_yield"]
    improvement_event = _event(
        change, FACTORS[2], improvement,
        ["FLOW_EVENT_TIME", "L4_BALANCE_EVENT_TIME", "PRIOR_FLOW_EVENT_TIME", "PRIOR_L4_BALANCE_EVENT_TIME"],
    )
    return pd.concat([q_event, ttm_event, improvement_event], ignore_index=True)


def prepare_events(events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    pieces = []
    for factor, group in events.groupby("factor", sort=False):
        available = assign_available_trade_date(group, calendar)
        available = available.sort_values(["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"])
        newest = available.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
        available = available.loc[available["QUARTER_INDEX"].eq(newest)]
        available = available.drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
        pieces.append(available[["SECURITY_ID", "AVAILABLE_DATE", "factor", "value"]])
    long = pd.concat(pieces, ignore_index=True)
    wide = long.pivot_table(index=["SECURITY_ID", "AVAILABLE_DATE"], columns="factor", values="value", aggfunc="last").reset_index()
    for factor in FACTORS:
        if factor not in wide:
            wide[factor] = np.nan
    wide = wide.sort_values(["SECURITY_ID", "AVAILABLE_DATE"])
    wide[FACTORS] = wide.groupby("SECURITY_ID", sort=False)[FACTORS].ffill()
    return wide[["SECURITY_ID", "AVAILABLE_DATE", *FACTORS]]


def generate(args: argparse.Namespace) -> None:
    dates = pd.read_parquet(args.panel, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    wide = prepare_events(calculate_events(args.pit_dir), calendar)
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
        panel = _normalize_panel(pd.read_parquet(args.panel, columns=KEYS, filters=filters))
        mapped = pd.merge_asof(
            panel.sort_values(["TRADE_DATE", "SECURITY_ID"]),
            wide.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]),
            by="SECURITY_ID", left_on="TRADE_DATE", right_on="AVAILABLE_DATE", direction="backward",
        )[KEYS + FACTORS]
        mapped[FACTORS] = mapped[FACTORS].astype("float32")
        chunks.append(mapped)
        for factor in FACTORS:
            coverage.append({"year": year, "factor": factor, "missing_rate": float(mapped[factor].isna().mean())})
        print(year, f"rows={len(mapped):,}")
    result = pd.concat(chunks).sort_values(KEYS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(args.coverage, index=False, encoding="utf-8-sig")


def fill(args: argparse.Namespace) -> None:
    tested = pd.read_csv(args.sparse_ic)
    if not set(FACTORS).issubset(set(tested["factor"].astype(str))):
        raise RuntimeError("all round-44 factors must be sparse-tested before filling")
    data = pd.read_parquet(args.sparse)
    before = data[FACTORS].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[FACTORS].transform("median")
    data[FACTORS] = data[FACTORS].fillna(medians).fillna(0.0).astype("float32")
    data.to_parquet(args.output, index=False, compression="zstd")
    pd.DataFrame({"factor": FACTORS, "missing_before_fill": before.reindex(FACTORS).values}).to_csv(args.report, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-sparse")
    g.add_argument("--panel", type=Path, required=True); g.add_argument("--pit-dir", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True); g.add_argument("--coverage", type=Path, required=True)
    f = sub.add_parser("fill-after-test")
    f.add_argument("--sparse", type=Path, required=True); f.add_argument("--sparse-ic", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True); f.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    generate(args) if args.command == "generate-sparse" else fill(args)


if __name__ == "__main__":
    main()
