"""Build analyst revision breadth, disagreement, and coverage-event factors.

The source table contains forecasts written on each date, rather than a
forward-filled consensus state.  Raw event observations are therefore tested
first and then accumulated with fixed half-lives.  No labels, ranks, or prior
successful factors enter construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twenty_third_alternative_event_factors import KEYS, prepare_available, security_mapping
from twenty_fourth_event_accumulation_factors import ewm_state


INDEXES = ["PROFIT", "EPS", "INCOME"]


def read_events(root: Path) -> pd.DataFrame:
    columns = [
        "SEC_CODE", "STAT_DATE", "FORE_YEAR", "FORE_QUARTER", "INDEX_NAME",
        "FORE_NUM", "FORE_AVE", "FORE_STA", "INCREA_ORG_NUM", "HOLD_ORG_NUM",
        "LOWER_ORG_NUM", "FIRST_ORG_NUM",
    ]
    pieces: list[pd.DataFrame] = []
    for path in sorted((root / "rr_profit_stat_d").glob("*.parquet")):
        frame = pd.read_parquet(
            path, columns=columns, filters=[("INDEX_NAME", "in", INDEXES)]
        )
        if not frame.empty:
            pieces.append(frame)
    if not pieces:
        raise FileNotFoundError("rr_profit_stat_d parquet files were not found")
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternative-root", type=Path, required=True)
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

    data = read_events(args.alternative_root)
    data["TICKER_SYMBOL"] = data["SEC_CODE"].astype(str).str.extract(r"(\d{6})", expand=False)
    data = data.merge(ticker, on="TICKER_SYMBOL", how="left", validate="many_to_one")
    data["EVENT_TIME"] = pd.to_datetime(data["STAT_DATE"], errors="coerce").dt.normalize()
    data["FORE_YEAR"] = pd.to_numeric(data["FORE_YEAR"], errors="coerce")
    data["HORIZON"] = data["FORE_YEAR"] - data["EVENT_TIME"].dt.year
    # FY0 and FY1 are standard near-term analyst horizons.  Older reported
    # years and long-range thin forecasts are excluded before any testing.
    data = data.loc[data["HORIZON"].isin([0, 1])].dropna(
        subset=["SECURITY_ID", "EVENT_TIME"]
    ).copy()
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    for column in [
        "FORE_NUM", "FORE_AVE", "FORE_STA", "INCREA_ORG_NUM",
        "HOLD_ORG_NUM", "LOWER_ORG_NUM", "FIRST_ORG_NUM",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    revised = data["INCREA_ORG_NUM"].fillna(0) + data["LOWER_ORG_NUM"].fillna(0)
    all_opinions = revised + data["HOLD_ORG_NUM"].fillna(0)
    data["BREADTH"] = (
        data["INCREA_ORG_NUM"].fillna(0) - data["LOWER_ORG_NUM"].fillna(0)
    ).div(all_opinions.clip(lower=1)).clip(-1, 1)
    data["REVISION_INTENSITY"] = revised.div(all_opinions.clip(lower=1)).clip(0, 1)
    data["DISPERSION"] = (
        data["FORE_STA"] / data["FORE_AVE"].abs().where(data["FORE_AVE"].abs().gt(1e-12))
    ).clip(0, 5)
    data["LOW_DISPERSION"] = -data["DISPERSION"]
    data["FIRST_COVERAGE"] = data["FIRST_ORG_NUM"].fillna(0).div(
        np.sqrt(data["FORE_NUM"].fillna(0).clip(lower=1))
    ).clip(0, 5)
    data["SUFFIX"] = data["INDEX_NAME"].str.lower() + "_fy" + data["HORIZON"].astype(int).astype(str)

    wide = data.pivot_table(
        index=["SECURITY_ID", "EVENT_TIME"], columns="SUFFIX",
        values=["BREADTH", "REVISION_INTENSITY", "LOW_DISPERSION", "FIRST_COVERAGE"],
        aggfunc="last",
    )
    wide.columns = [f"{metric.lower()}_{suffix}" for metric, suffix in wide.columns]
    wide = wide.reset_index()
    for horizon in [0, 1]:
        fields = [f"breadth_{name}_fy{horizon}" for name in ["profit", "eps", "income"]]
        for field in fields:
            if field not in wide:
                wide[field] = np.nan
        wide[f"confirmation_fy{horizon}"] = wide[fields].mean(axis=1)
        wide[f"margin_revision_fy{horizon}"] = (
            wide[f"breadth_profit_fy{horizon}"] - wide[f"breadth_income_fy{horizon}"]
        )
    source_columns = [
        "breadth_profit_fy0", "breadth_profit_fy1", "breadth_eps_fy0",
        "breadth_income_fy0", "confirmation_fy0", "confirmation_fy1",
        "margin_revision_fy0", "low_dispersion_profit_fy0",
        "first_coverage_profit_fy0", "revision_intensity_profit_fy0",
    ]
    for column in source_columns:
        if column not in wide:
            wide[column] = np.nan
    sparse = prepare_available(wide, calendar, source_columns)
    sparse.to_parquet(output / "round29_sparse_analyst_events.parquet", index=False)

    dense = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    dense = dense.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(drop=True)
    specifications = [
        ("r29_profit_breadth_fy0_hl10", "breadth_profit_fy0", 10),
        ("r29_profit_breadth_fy0_hl20", "breadth_profit_fy0", 20),
        ("r29_profit_breadth_fy1_hl20", "breadth_profit_fy1", 20),
        ("r29_eps_breadth_fy0_hl20", "breadth_eps_fy0", 20),
        ("r29_income_breadth_fy0_hl20", "breadth_income_fy0", 20),
        ("r29_confirmation_fy0_hl10", "confirmation_fy0", 10),
        ("r29_confirmation_fy0_hl20", "confirmation_fy0", 20),
        ("r29_confirmation_fy1_hl20", "confirmation_fy1", 20),
        ("r29_margin_revision_fy0_hl20", "margin_revision_fy0", 20),
        ("r29_low_dispersion_profit_fy0_hl20", "low_dispersion_profit_fy0", 20),
        ("r29_first_coverage_profit_fy0_hl20", "first_coverage_profit_fy0", 20),
        ("r29_revision_intensity_profit_fy0_hl20", "revision_intensity_profit_fy0", 20),
    ]
    result = dense[KEYS].copy()
    for name, source, half_life in specifications:
        result[name] = ewm_state(dense, source, half_life)
        print(name, flush=True)
    result.to_parquet(output / "round29_analyst_breadth_factors.parquet", index=False)
    metadata = {
        "source_rows": len(data),
        "event_security_dates": len(wide),
        "sparse_rows": len(sparse),
        "factor_columns": [name for name, _, _ in specifications],
        "uses_rank": False,
        "uses_label": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
