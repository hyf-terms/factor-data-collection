"""Build daily consensus-state revision factors from selected core forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twenty_third_alternative_event_factors import KEYS, prepare_available, security_mapping


VALUE_COLUMNS = ["CON_PROFIT", "CON_INCOME", "CON_OCF", "CON_DIV"]


def symmetric_delta(current: pd.Series, previous: pd.Series) -> pd.Series:
    denominator = current.abs() + previous.abs()
    return (2 * (current - previous)).div(denominator.where(denominator.gt(1e-12))).clip(-2, 2)


def read_near_horizons(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    columns = ["SEC_CODE", "REP_FORE_DATE", "FORE_YEAR", *VALUE_COLUMNS]
    for path in sorted((root / "con_sec_coredata_2").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame["REP_FORE_DATE"] = pd.to_datetime(frame["REP_FORE_DATE"], errors="coerce").dt.normalize()
        frame["FORE_YEAR"] = pd.to_numeric(frame["FORE_YEAR"], errors="coerce")
        frame["HORIZON"] = frame["FORE_YEAR"] - frame["REP_FORE_DATE"].dt.year
        frame = frame.loc[frame["HORIZON"].isin([0, 1])].copy()
        if not frame.empty:
            pieces.append(frame)
    if not pieces:
        raise FileNotFoundError("selected con_sec_coredata_2 parquet files were not found")
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus-root", type=Path, required=True)
    parser.add_argument("--pit-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel, columns=KEYS)
    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
    panel["SECURITY_ID"] = panel["SECURITY_ID"].astype("int64")
    panel = panel.drop_duplicates(KEYS)
    calendar = pd.DatetimeIndex(panel["TRADE_DATE"].unique()).sort_values()
    _, ticker = security_mapping(args.pit_root)

    data = read_near_horizons(args.consensus_root)
    data["TICKER_SYMBOL"] = data["SEC_CODE"].astype(str).str.extract(r"(\d{6})", expand=False)
    data = data.merge(ticker, on="TICKER_SYMBOL", how="left", validate="many_to_one")
    data = data.dropna(subset=["SECURITY_ID", "REP_FORE_DATE"]).copy()
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    for column in VALUE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    wide = data.pivot_table(
        index=["SECURITY_ID", "REP_FORE_DATE"], columns="HORIZON",
        values=VALUE_COLUMNS, aggfunc="last",
    )
    wide.columns = [f"{name.lower()}_fy{int(horizon)}" for name, horizon in wide.columns]
    wide = wide.reset_index().sort_values(["SECURITY_ID", "REP_FORE_DATE"])
    for column in [f"{name.lower()}_fy{h}" for name in VALUE_COLUMNS for h in [0, 1]]:
        if column not in wide:
            wide[column] = np.nan
    wide["margin_fy1"] = wide["con_profit_fy1"].div(wide["con_income_fy1"].abs().where(wide["con_income_fy1"].abs().gt(1e-12)))
    wide["cash_conversion_fy1"] = wide["con_ocf_fy1"].div(wide["con_profit_fy1"].abs().where(wide["con_profit_fy1"].abs().gt(1e-12))).clip(-10, 10)
    wide["profit_growth_expected"] = symmetric_delta(wide["con_profit_fy1"], wide["con_profit_fy0"])

    groups = wide.groupby("SECURITY_ID", sort=False)
    source_columns: list[str] = []
    for lag in [5, 20]:
        for metric in ["con_profit_fy1", "con_income_fy1", "con_ocf_fy1", "con_div_fy1"]:
            name = f"{metric}_revision_{lag}d"
            wide[name] = symmetric_delta(wide[metric], groups[metric].shift(lag))
            source_columns.append(name)
        for metric in ["margin_fy1", "cash_conversion_fy1", "profit_growth_expected"]:
            name = f"{metric}_change_{lag}d"
            wide[name] = (wide[metric] - groups[metric].shift(lag)).clip(-5, 5)
            source_columns.append(name)
        wide[f"confirmation_{lag}d"] = wide[[
            f"con_profit_fy1_revision_{lag}d",
            f"con_income_fy1_revision_{lag}d",
            f"con_ocf_fy1_revision_{lag}d",
        ]].mean(axis=1)
        source_columns.append(f"confirmation_{lag}d")

    events = wide.rename(columns={"REP_FORE_DATE": "EVENT_TIME"})
    sparse = prepare_available(events, calendar, source_columns)
    sparse.to_parquet(output / "round30_sparse_consensus_revisions.parquet", index=False)
    dense = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    result = dense[KEYS].copy()
    factor_columns: list[str] = []
    for source in source_columns:
        name = "r30_" + source
        result[name] = pd.to_numeric(dense[source], errors="coerce").fillna(0).astype("float32")
        factor_columns.append(name)
    result.to_parquet(output / "round30_consensus_state_revision_factors.parquet", index=False)
    metadata = {
        "near_horizon_rows": len(data),
        "consensus_security_dates": len(wide),
        "factor_columns": factor_columns,
        "uses_rank": False,
        "uses_label": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
