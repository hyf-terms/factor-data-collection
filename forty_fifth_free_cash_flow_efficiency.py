"""Round 45: PIT free-cash-flow capital efficiency, without earnings inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from ninth_round_independent_factors import _lag_table, _latest_time, _safe_ratio, build_balance_events
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS, build_standalone_quarterly_metric


FACTORS = [
    "r45_ttm_fcf_assets",
    "r45_ttm_fcf_invested_capital",
    "r45_ttm_capex_self_financing",
]


def events(pit_dir: Path) -> pd.DataFrame:
    cash = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=COMMON_COLUMNS + ["N_CF_OPERATE_A", "PUR_FIX_ASSETS_OTH"],
        engine="pyarrow",
    )
    cash["N_CF_OPERATE_A"] = pd.to_numeric(cash["N_CF_OPERATE_A"], errors="coerce")
    cash["PUR_FIX_ASSETS_OTH"] = pd.to_numeric(cash["PUR_FIX_ASSETS_OTH"], errors="coerce").fillna(0.0)
    flows = {
        field: build_standalone_quarterly_metric(cash, field, name="cashflow PIT")
        for field in ["N_CF_OPERATE_A", "PUR_FIX_ASSETS_OTH"]
    }
    balance_fields = [
        "INDUSTRY_CATEGORY", "T_ASSETS", "CASH_C_EQUIV", "ST_BORR", "NCL_WITHIN_1_Y",
        "LT_BORR", "BOND_PAYABLE", "PAID_IN_CAPITAL", "T_EQUITY_ATTR_P",
        "RETAINED_EARNINGS", "INTAN_ASSETS", "GOODWILL",
    ]
    raw = pd.read_parquet(
        pit_dir / "new_pit_balance", columns=COMMON_COLUMNS + balance_fields, engine="pyarrow"
    )
    balance = build_balance_events(raw)
    ttm = _merge_flow_tables(flows, ["N_CF_OPERATE_A", "PUR_FIX_ASSETS_OTH"], ttm=True)
    lag = _lag_table(
        balance, 4, "L4",
        ["BALANCE_EVENT_TIME", "T_ASSETS", "CASH_C_EQUIV", "ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE", "T_EQUITY_ATTR_P"],
    )
    data = ttm.merge(lag, on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    cfo = data["TTM_N_CF_OPERATE_A"]
    capex = data["TTM_PUR_FIX_ASSETS_OTH"].clip(lower=0.0)
    fcf = cfo - capex
    debt = data[["L4_ST_BORR", "L4_NCL_WITHIN_1_Y", "L4_LT_BORR", "L4_BOND_PAYABLE"]].sum(axis=1)
    invested = data["L4_T_EQUITY_ATTR_P"] + debt - data["L4_CASH_C_EQUIV"]
    scale = data["L4_T_ASSETS"].abs().where(data["L4_T_ASSETS"].abs().gt(0))
    values = {
        FACTORS[0]: _safe_ratio(fcf, data["L4_T_ASSETS"]),
        FACTORS[1]: _safe_ratio(fcf, invested),
        FACTORS[2]: _safe_ratio(cfo - capex, capex + 0.01 * scale),
    }
    pieces = []
    for factor, value in values.items():
        out = data[["SECURITY_ID", "QUARTER_INDEX"]].copy()
        out["EVENT_TIME"] = _latest_time(data, ["FLOW_EVENT_TIME", "L4_BALANCE_EVENT_TIME"])
        out["factor"] = factor
        out["value"] = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(-20, 20)
        pieces.append(out.dropna(subset=["EVENT_TIME", "value"]))
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--pit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    dates = pd.read_parquet(args.panel, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    raw = events(args.pit_dir)
    pieces = []
    for factor, group in raw.groupby("factor", sort=False):
        a = assign_available_trade_date(group, calendar).sort_values(["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"])
        newest = a.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
        a = a.loc[a["QUARTER_INDEX"].eq(newest)].drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
        pieces.append(a[["SECURITY_ID", "AVAILABLE_DATE", "factor", "value"]])
    wide = pd.concat(pieces).pivot_table(index=["SECURITY_ID", "AVAILABLE_DATE"], columns="factor", values="value", aggfunc="last").reset_index()
    wide = wide.sort_values(["SECURITY_ID", "AVAILABLE_DATE"])
    wide[FACTORS] = wide.groupby("SECURITY_ID", sort=False)[FACTORS].ffill()
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        filters=[("TRADE_DATE",">=",pd.Timestamp(year,1,1)),("TRADE_DATE","<=",pd.Timestamp(year,12,31))]
        panel = _normalize_panel(pd.read_parquet(args.panel, columns=KEYS, filters=filters))
        mapped = pd.merge_asof(panel.sort_values(["TRADE_DATE","SECURITY_ID"]), wide.sort_values(["AVAILABLE_DATE","SECURITY_ID"]), by="SECURITY_ID", left_on="TRADE_DATE", right_on="AVAILABLE_DATE", direction="backward")[KEYS+FACTORS]
        mapped[FACTORS] = mapped[FACTORS].astype("float32")
        chunks.append(mapped)
        for factor in FACTORS: coverage.append({"year":year,"factor":factor,"missing_rate":float(mapped[factor].isna().mean())})
        print(year, f"rows={len(mapped):,}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(chunks).sort_values(KEYS).to_parquet(args.output,index=False,compression="zstd")
    pd.DataFrame(coverage).to_csv(args.coverage,index=False,encoding="utf-8-sig")


if __name__ == "__main__":
    main()
