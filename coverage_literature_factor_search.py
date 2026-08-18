"""Build high-coverage literature/vendor financial factors without ranks.

The source is the PIT single-quarter financial-indicator table.  Every
candidate must have at least 90% source-event coverage in its applicable
sample.  Values are winsorized by date at 1%/99%, but are never converted to
cross-sectional ranks and no median/quantile company filter is applied.
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
from pead_sue_factor import assign_available_trade_date
from quarterly_indicator_factor_search import _winsorize_daily


BASE_DIR = Path(__file__).resolve().parent
QUARTERLY_WINDOW = 120
ANNUAL_WINDOW = 252
MIN_SOURCE_COVERAGE = 0.90

DIRECT_FIELDS = {
    "eps_yoy": "EPS_YOY",
    "eps_cut_yoy": "EPS_CUT_YOY",
    "gross_profit_yoy": "GROSS_PROFIT_YOY",
    "revenue_yoy": "REVENUE_YOY",
    "oper_profit_yoy": "OPER_PROFIT_YOY",
    "parent_profit_yoy": "NI_ATTR_P_YOY",
    "roe_yoy": "ROE_YOY",
    "np_margin_yoy": "NP_MARGIN_YOY",
    "cfo_yoy": "N_CF_OPA_YOY",
    "roa": "ROA",
    "roe": "ROE",
    "cost_exp_profit": "P_COST_EXP",
    "cash_asset_recovery": "C_RCVRY_A",
    "deducted_profit_share": "NI_CUT_NI",
    "operating_margin": "OP_TR",
    "cash_profit_conversion": "N_CF_OPA_NIA",
}

COMPOSITES = {
    "profit_growth_breadth": [
        "OPER_PROFIT_YOY",
        "NI_ATTR_P_YOY",
        "T_PROFIT_YOY",
    ],
    "per_share_earnings_growth": ["EPS_YOY", "EPS_CUT_YOY"],
    "profitability_momentum": [
        "ROE_YOY",
        "NP_MARGIN_YOY",
        "OPER_PROFIT_YOY",
    ],
    "profit_cash_confirmation": [
        "NI_ATTR_P_YOY",
        "GROSS_PROFIT_YOY",
        "N_CF_OPA_YOY",
    ],
    "profitability_level": ["ROA", "ROE", "OP_TR"],
}

VALUE_FIELDS = sorted(
    set(DIRECT_FIELDS.values()).union(*map(set, COMPOSITES.values()))
)
SIGNAL_ALIASES = [*DIRECT_FIELDS, *COMPOSITES]
CANDIDATE_COLUMNS = [
    *[f"latestq_vendor_{name}_120d" for name in SIGNAL_ALIASES],
    *[f"annual_vendor_{name}_252d" for name in SIGNAL_ALIASES],
]

LITERATURE_BASIS = {
    "roe": "Hou-Xue-Zhang return-on-equity profitability signal",
    "roa": "Piotroski profitability signal",
    "gross_profit_yoy": "Novy-Marx gross-profitability family",
    "eps_yoy": "Chan-Jegadeesh-Lakonishok earnings momentum",
    "eps_cut_yoy": "PEAD/earnings-momentum family; vendor-defined EPS growth",
    "cash_asset_recovery": "Ball et al. cash-based profitability family",
}


def load_events(dataset: Path) -> pd.DataFrame:
    columns = [
        "SECURITY_ID",
        "ID",
        "PUBLISH_DATE",
        "END_DATE_REP",
        "END_DATE",
        "UPDATE_TIME",
        "REPORT_TYPE",
        *VALUE_FIELDS,
    ]
    events = pd.read_parquet(dataset, columns=columns)
    for column in [
        "PUBLISH_DATE",
        "END_DATE_REP",
        "END_DATE",
        "UPDATE_TIME",
    ]:
        events[column] = pd.to_datetime(events[column], errors="coerce")
    events = events.loc[
        events["END_DATE"].eq(events["END_DATE_REP"])
        & events["REPORT_TYPE"].isin(["Q1", "S1", "Q3", "A"])
    ].copy()
    quarter_map = {"Q1": 1, "S1": 2, "Q3": 3, "A": 4}
    events["FISCAL_QUARTER"] = events["REPORT_TYPE"].map(quarter_map)
    events["FISCAL_YEAR"] = events["END_DATE"].dt.year
    events["QUARTER_INDEX"] = (
        events["FISCAL_YEAR"] * 4 + events["FISCAL_QUARTER"]
    )
    events["EVENT_TIME"] = (
        events["PUBLISH_DATE"].dt.normalize()
        + pd.to_timedelta(86_399, unit="s")
    )
    events = events.dropna(
        subset=[
            "SECURITY_ID",
            "ID",
            "EVENT_TIME",
            "FISCAL_QUARTER",
            "QUARTER_INDEX",
        ]
    ).copy()
    events["SECURITY_ID"] = pd.to_numeric(
        events["SECURITY_ID"], errors="raise"
    ).astype("int64")
    events["ID"] = pd.to_numeric(events["ID"], errors="raise").astype(
        "int64"
    )
    events["FISCAL_QUARTER"] = events["FISCAL_QUARTER"].astype("int8")
    events["QUARTER_INDEX"] = events["QUARTER_INDEX"].astype("int32")
    for column in VALUE_FIELDS:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    events = events.sort_values(
        [
            "SECURITY_ID",
            "PUBLISH_DATE",
            "QUARTER_INDEX",
            "UPDATE_TIME",
            "ID",
        ],
        na_position="first",
    ).drop_duplicates(
        ["SECURITY_ID", "PUBLISH_DATE", "QUARTER_INDEX"],
        keep="last",
    )
    return events.reset_index(drop=True)


def source_coverage(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = {
        "latest_quarterly": events,
        "annual": events.loc[events["FISCAL_QUARTER"].eq(4)],
    }
    specs = {
        **{name: [field] for name, field in DIRECT_FIELDS.items()},
        **COMPOSITES,
    }
    for scope, sample in scopes.items():
        for name, fields in specs.items():
            complete = sample[fields].notna().all(axis=1)
            rows.append(
                {
                    "scope": scope,
                    "signal": name,
                    "fields": ";".join(fields),
                    "source_rows": len(sample),
                    "complete_rows": int(complete.sum()),
                    "coverage": float(complete.mean()),
                    "missing_rate": float(1.0 - complete.mean()),
                    "uses_rank": False,
                    "company_filter": "none",
                    "literature_basis": LITERATURE_BASIS.get(
                        name, "Tonglian-defined financial indicator/composite"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    failed = result.loc[result["coverage"].lt(MIN_SOURCE_COVERAGE)]
    if not failed.empty:
        details = failed[["scope", "signal", "coverage"]].to_dict(
            orient="records"
        )
        raise RuntimeError(
            f"Candidates below {MIN_SOURCE_COVERAGE:.0%} source coverage: "
            f"{details}"
        )
    return result


def prepare_available_events(
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    available = assign_available_trade_date(events, calendar)
    available = available.sort_values(
        [
            "SECURITY_ID",
            "AVAILABLE_DATE",
            "EVENT_TIME",
            "QUARTER_INDEX",
            "ID",
        ]
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
            *VALUE_FIELDS,
        ]
    ].reset_index(drop=True)


def map_events_to_panel(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    calendar_ordinal: pd.Series,
) -> pd.DataFrame:
    joined = pd.merge_asof(
        panel.sort_values(["TRADE_DATE", "SECURITY_ID"]),
        events.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]),
        by="SECURITY_ID",
        left_on="TRADE_DATE",
        right_on="AVAILABLE_DATE",
        direction="backward",
    )
    joined["EVENT_AGE"] = (
        joined["TRADE_DATE"].map(calendar_ordinal)
        - joined["AVAILABLE_DATE"].map(calendar_ordinal)
    )
    return joined.sort_values(KEYS).reset_index(drop=True)


def _signals(data: pd.DataFrame) -> dict[str, pd.Series]:
    values = data.copy()
    values[VALUE_FIELDS] = _winsorize_daily(values, VALUE_FIELDS)
    signals: dict[str, pd.Series] = {
        name: values[field] for name, field in DIRECT_FIELDS.items()
    }
    for name, fields in COMPOSITES.items():
        signals[name] = values[fields].mean(axis=1, skipna=False)
    return signals


def build_candidates(
    latest_quarterly: pd.DataFrame,
    annual: pd.DataFrame,
) -> pd.DataFrame:
    latest_signals = _signals(latest_quarterly)
    annual_signals = _signals(annual)
    quarterly_valid = latest_quarterly["EVENT_AGE"].ge(0) & (
        latest_quarterly["EVENT_AGE"].lt(QUARTERLY_WINDOW)
    )
    annual_valid = annual["EVENT_AGE"].ge(0) & annual["EVENT_AGE"].lt(
        ANNUAL_WINDOW
    )
    result = latest_quarterly[KEYS].copy()
    for name in SIGNAL_ALIASES:
        result[f"latestq_vendor_{name}_120d"] = latest_signals[name].where(
            quarterly_valid
        )
        result[f"annual_vendor_{name}_252d"] = annual_signals[name].where(
            annual_valid
        )
    for column in CANDIDATE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("float32")
    return result[KEYS + CANDIDATE_COLUMNS]


def generate(
    factor_path: Path,
    indicator_dataset: Path,
    output_path: Path,
    coverage_path: Path,
    metadata_path: Path,
) -> None:
    dates = pd.read_parquet(factor_path, columns=["TRADE_DATE"])[
        "TRADE_DATE"
    ]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique())
    calendar = calendar.normalize().drop_duplicates().sort_values()
    calendar_ordinal = pd.Series(
        np.arange(len(calendar), dtype=np.int16), index=calendar
    )
    raw_events = load_events(indicator_dataset)
    coverage = source_coverage(raw_events)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    events = prepare_available_events(raw_events, calendar)
    annual_events = prepare_available_events(
        raw_events.loc[raw_events["FISCAL_QUARTER"].eq(4)].copy(),
        calendar,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    mapped_non_null = {name: 0 for name in CANDIDATE_COLUMNS}
    total_rows = 0
    try:
        for year in sorted(set(calendar.year)):
            panel = pd.read_parquet(
                factor_path,
                columns=KEYS,
                filters=[
                    ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
                    ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
                ],
            )
            panel = _normalize_panel(panel)
            mapped = map_events_to_panel(panel, events, calendar_ordinal)
            mapped_annual = map_events_to_panel(
                panel, annual_events, calendar_ordinal
            )
            candidates = build_candidates(mapped, mapped_annual)
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd"
                )
            writer.write_table(table)
            total_rows += len(candidates)
            counts = candidates[CANDIDATE_COLUMNS].notna().sum()
            for name, count in counts.items():
                mapped_non_null[name] += int(count)
            print(f"{year}: rows={len(candidates):,}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No candidates were generated")
    os.replace(temporary, output_path)
    metadata = {
        "candidate_count": len(CANDIDATE_COLUMNS),
        "source_event_count": len(raw_events),
        "available_event_count": len(events),
        "available_annual_event_count": len(annual_events),
        "panel_rows": total_rows,
        "quarterly_window_trading_days": QUARTERLY_WINDOW,
        "annual_window_trading_days": ANNUAL_WINDOW,
        "minimum_source_coverage": MIN_SOURCE_COVERAGE,
        "uses_cross_sectional_rank": False,
        "winsorization": "daily 1%/99%",
        "company_filter": "none",
        "candidate_non_null_rows": mapped_non_null,
        "output": str(output_path),
        "coverage_file": str(coverage_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote: {output_path}")
    print(f"Coverage: {coverage_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factors", type=Path, default=BASE_DIR / "factors.parquet"
    )
    parser.add_argument(
        "--indicator-pit",
        type=Path,
        default=(
            BASE_DIR
            / "data"
            / "quarterly_financial_indicators"
            / "quarterly_financial_indicator_pit"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "coverage_literature_no_rank_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "coverage_literature_no_rank_coverage.csv"
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "coverage_literature_no_rank_metadata.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        factor_path=args.factors.resolve(),
        indicator_dataset=args.indicator_pit.resolve(),
        output_path=args.output.resolve(),
        coverage_path=args.coverage_output.resolve(),
        metadata_path=args.metadata_output.resolve(),
    )
