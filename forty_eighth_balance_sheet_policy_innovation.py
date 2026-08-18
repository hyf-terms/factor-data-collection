"""Round 48: PIT balance-sheet policy innovations, without profit fields.

Each quarterly stock is first converted to a same-quarter year-on-year change
scaled by beginning assets.  The change is then standardized by the firm's
own previous eight seasonal changes (minimum four).  The current observation
is never included in its own mean or volatility estimate.  Signals cover cash
accumulation, debt reduction and low paid-in-capital issuance; their fixed
equal-weight composite is also emitted.  No labels or percentile ranks enter
construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from ninth_round_independent_factors import BALANCE_FIELDS, build_balance_events, _lag_table
from quarterly_f_score import COMMON_COLUMNS


FACTORS = [
    "r48_cash_accumulation_innovation",
    "r48_debt_reduction_innovation",
    "r48_low_equity_issuance_innovation",
    "r48_financial_policy_innovation_composite",
]


def _safe_scale(x: pd.Series, assets: pd.Series) -> pd.Series:
    floor = assets.abs().median(skipna=True) * 1e-8
    return x.div(assets.where(assets.abs().gt(max(float(floor), 1e-12))))


def _prior_standardize(frame: pd.DataFrame, column: str) -> pd.Series:
    group = frame.groupby("SECURITY_ID", sort=False)[column]
    prior_mean = group.transform(lambda x: x.shift(1).rolling(8, min_periods=4).mean())
    prior_std = group.transform(lambda x: x.shift(1).rolling(8, min_periods=4).std())
    return (frame[column] - prior_mean).div(prior_std.where(prior_std.gt(1e-8)))


def build_events(pit_dir: Path) -> pd.DataFrame:
    raw = pd.read_parquet(
        pit_dir / "new_pit_balance", columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    bal = build_balance_events(raw)
    lag = _lag_table(
        bal, 4, "L4",
        ["BALANCE_EVENT_TIME", "T_ASSETS", "CASH_C_EQUIV", "ST_BORR",
         "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE", "PAID_IN_CAPITAL"],
    )
    d = bal.merge(lag, on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    d = d.sort_values(["SECURITY_ID", "QUARTER_INDEX"]).copy()
    debt = d[["ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE"]].sum(axis=1)
    lag_debt = d[["L4_ST_BORR", "L4_NCL_WITHIN_1_Y", "L4_LT_BORR", "L4_BOND_PAYABLE"]].sum(axis=1)
    d["cash_change"] = _safe_scale(d["CASH_C_EQUIV"] - d["L4_CASH_C_EQUIV"], d["L4_T_ASSETS"])
    d["debt_reduction"] = -_safe_scale(debt - lag_debt, d["L4_T_ASSETS"])
    d["low_equity_issuance"] = -_safe_scale(d["PAID_IN_CAPITAL"] - d["L4_PAID_IN_CAPITAL"], d["L4_T_ASSETS"])
    d[FACTORS[0]] = _prior_standardize(d, "cash_change")
    d[FACTORS[1]] = _prior_standardize(d, "debt_reduction")
    d[FACTORS[2]] = _prior_standardize(d, "low_equity_issuance")
    # Fixed equal weights. Require all three legs in the sparse diagnostic.
    d[FACTORS[3]] = d[FACTORS[:3]].mean(axis=1, skipna=False)
    d["EVENT_TIME"] = pd.concat(
        [pd.to_datetime(d["BALANCE_EVENT_TIME"]), pd.to_datetime(d["L4_BALANCE_EVENT_TIME"])],
        axis=1,
    ).max(axis=1)
    return d[["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", *FACTORS]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--pit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    calendar = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(args.panel, columns=["TRADE_DATE"])["TRADE_DATE"]).unique()
    ).normalize().sort_values()
    events = build_events(args.pit_dir)
    pieces = []
    for factor in FACTORS:
        e = events[["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", factor]].dropna(subset=[factor]).rename(columns={factor: "value"})
        a = assign_available_trade_date(e, calendar).sort_values(["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"])
        newest = a.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
        a = a.loc[a["QUARTER_INDEX"].eq(newest)].drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
        pieces.append(a[["SECURITY_ID", "AVAILABLE_DATE", "value"]].rename(columns={"value": factor}))
    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.merge(piece, on=["SECURITY_ID", "AVAILABLE_DATE"], how="outer")
    wide = wide.sort_values(["SECURITY_ID", "AVAILABLE_DATE"])
    wide[FACTORS] = wide.groupby("SECURITY_ID", sort=False)[FACTORS].ffill()
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        panel = _normalize_panel(pd.read_parquet(
            args.panel, columns=KEYS,
            filters=[("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))],
        ))
        mapped = pd.merge_asof(
            panel.sort_values(["TRADE_DATE", "SECURITY_ID"]),
            wide.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]),
            by="SECURITY_ID", left_on="TRADE_DATE", right_on="AVAILABLE_DATE", direction="backward",
        )[KEYS + FACTORS]
        mapped[FACTORS] = mapped[FACTORS].replace([np.inf, -np.inf], np.nan).astype("float32")
        chunks.append(mapped)
        coverage.extend({"year": year, "factor": f, "missing_rate": float(mapped[f].isna().mean())} for f in FACTORS)
        print(year, f"rows={len(mapped):,}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(chunks).sort_values(KEYS).to_parquet(args.output, index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(args.coverage, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
