"""Rounds 56--59: dense temporal and structural financial candidates.

The raw event panel is deliberately kept sparse for the first diagnostic.
Only after that diagnostic may ``fill-after-test`` replace unavailable company
history with the same-date cross-sectional median.  No percentile rank, label
fit, Q1 restriction, or fixed post-announcement window is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eighth_round_literature_factors as workflow
from event_financial_factor_search import KEYS, _normalize_panel
from fifteenth_round_temporal_financial_factors import _read_inputs


CANDIDATE_COLUMNS = [
    "r56_cfoa_slope8",
    "r56_revenue_assets_slope8",
    "r56_cash_support_slope8",
    "r56_nonprofit_trend_equal3",
    "r57_cash_earnings_yoy_gap_mean4",
    "r57_cash_earnings_gap_convergence",
    "r57_cash_earnings_state_equal2",
    "r58_financing_headroom_state",
    "r58_financing_headroom_change4",
    "r58_constraint_transition_equal3",
    "r59_industry_financial_comparability4",
    "r59_peer_cashflow_diffusion_gap",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    return pd.to_numeric(numerator, errors="coerce").div(
        denominator.abs().where(denominator.abs().gt(1e-12))
    ).replace([np.inf, -np.inf], np.nan)


def _rolling_slope(values: pd.Series, window: int = 8, minimum: int = 6) -> pd.Series:
    def slope(array: np.ndarray) -> float:
        valid = np.isfinite(array)
        if valid.sum() < minimum:
            return np.nan
        x = np.arange(len(array), dtype="float64")[valid]
        y = array[valid]
        x = x - x.mean()
        denominator = float(np.dot(x, x))
        return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0 else np.nan

    return values.rolling(window, min_periods=minimum).apply(slope, raw=True)


def _robust_group_z(values: pd.Series, groups: list[pd.Series]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    center = values.groupby(groups, sort=False).transform("median")
    deviation = (values - center).abs()
    mad = deviation.groupby(groups, sort=False).transform("median")
    z = (values - center).div((1.4826 * mad).where(mad.gt(1e-12)))
    return z.replace([np.inf, -np.inf], np.nan).clip(-8.0, 8.0)


def _flow_table(flows: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    return flows[field][["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", field]].rename(
        columns={"EVENT_TIME": f"{field}_EVENT_TIME"}
    )


def build_quarterly_states(pit_dir: Path) -> pd.DataFrame:
    flows, balance = _read_inputs(pit_dir)
    base = balance.copy()
    for field in ["N_INCOME_ATTR_P", "REVENUE", "COGS", "N_CF_OPERATE_A"]:
        base = base.merge(
            _flow_table(flows, field),
            on=["SECURITY_ID", "QUARTER_INDEX"],
            how="left",
            validate="one_to_one",
        )

    pieces: list[pd.DataFrame] = []
    for security_id, group in base.groupby("SECURITY_ID", sort=False):
        ordered = group.sort_values("QUARTER_INDEX")
        full_index = pd.RangeIndex(
            int(ordered["QUARTER_INDEX"].min()), int(ordered["QUARTER_INDEX"].max()) + 1
        )
        x = ordered.set_index("QUARTER_INDEX").reindex(full_index)
        assets = pd.to_numeric(x["T_ASSETS"], errors="coerce")
        lag_assets = assets.shift(1)
        ni = pd.to_numeric(x["N_INCOME_ATTR_P"], errors="coerce")
        revenue = pd.to_numeric(x["REVENUE"], errors="coerce")
        cfo = pd.to_numeric(x["N_CF_OPERATE_A"], errors="coerce")
        cfoa = _safe_ratio(cfo, lag_assets)
        revenue_assets = _safe_ratio(revenue, lag_assets)
        cash_support = _safe_ratio(cfo - ni, lag_assets)
        cfo_yoy = _safe_ratio(cfo - cfo.shift(4), lag_assets)
        ni_yoy = _safe_ratio(ni - ni.shift(4), lag_assets)
        cash_earnings_gap = cfo_yoy - ni_yoy

        cash = pd.to_numeric(x["CASH_C_EQUIV"], errors="coerce")
        short_debt = (
            pd.to_numeric(x["ST_BORR"], errors="coerce").fillna(0.0)
            + pd.to_numeric(x["NCL_WITHIN_1_Y"], errors="coerce").fillna(0.0)
        )
        cfo_ttm = cfo.rolling(4, min_periods=4).sum()
        headroom = _safe_ratio(cash + cfo_ttm - short_debt, assets)
        low_liability_growth = -_safe_ratio(
            pd.to_numeric(x["T_LIAB"], errors="coerce")
            - pd.to_numeric(x["T_LIAB"], errors="coerce").shift(4),
            lag_assets,
        )

        out = x.copy()
        out["SECURITY_ID"] = security_id
        out["QUARTER_INDEX"] = full_index
        out["cfoa"] = cfoa
        out["revenue_assets"] = revenue_assets
        out["cash_support"] = cash_support
        out["cfo_yoy"] = cfo_yoy
        out["cfoa_slope8"] = _rolling_slope(cfoa)
        out["revenue_assets_slope8"] = _rolling_slope(revenue_assets)
        out["cash_support_slope8"] = _rolling_slope(cash_support)
        out["cash_earnings_yoy_gap_mean4"] = cash_earnings_gap.rolling(4, min_periods=3).mean()
        out["cash_earnings_gap_convergence"] = cash_earnings_gap.abs().shift(1) - cash_earnings_gap.abs()
        out["headroom"] = headroom
        out["headroom_change4"] = headroom - headroom.shift(4)
        out["low_liability_growth"] = low_liability_growth
        pieces.append(out.loc[ordered["QUARTER_INDEX"].to_numpy()])

    result = pd.concat(pieces, ignore_index=True)
    time_columns = [column for column in result if column.endswith("EVENT_TIME")]
    result["EVENT_TIME"] = pd.concat(
        [pd.to_datetime(result[column], errors="coerce") for column in time_columns], axis=1
    ).max(axis=1)

    quarter = result["QUARTER_INDEX"]
    z_columns = [
        "cfoa_slope8", "revenue_assets_slope8", "cash_support_slope8",
        "cash_earnings_yoy_gap_mean4", "cash_earnings_gap_convergence",
        "headroom", "headroom_change4", "low_liability_growth",
    ]
    for column in z_columns:
        result[f"z_{column}"] = _robust_group_z(result[column], [quarter])
    result["nonprofit_trend_equal3"] = result[
        ["z_cfoa_slope8", "z_revenue_assets_slope8", "z_cash_support_slope8"]
    ].mean(axis=1, skipna=False)
    result["cash_earnings_state_equal2"] = result[
        ["z_cash_earnings_yoy_gap_mean4", "z_cash_earnings_gap_convergence"]
    ].mean(axis=1, skipna=False)
    result["constraint_transition_equal3"] = result[
        ["z_headroom", "z_headroom_change4", "z_low_liability_growth"]
    ].mean(axis=1, skipna=False)

    industry = result["INDUSTRY_CATEGORY"].astype("string").fillna("UNKNOWN")
    financial_features = ["cfoa", "revenue_assets", "cash_support", "headroom"]
    industry_z = pd.DataFrame(index=result.index)
    for column in financial_features:
        industry_z[column] = _robust_group_z(result[column], [quarter, industry])
    result["industry_financial_comparability4"] = -np.sqrt(
        industry_z.pow(2).mean(axis=1, skipna=False)
    )
    peer_cfo = result["cfo_yoy"].groupby([quarter, industry], sort=False).transform("median")
    prior_own_cfo = result.sort_values(["SECURITY_ID", "QUARTER_INDEX"]).groupby(
        "SECURITY_ID", sort=False
    )["cfo_yoy"].shift(1).reindex(result.index)
    result["peer_cashflow_diffusion_gap"] = peer_cfo - prior_own_cfo
    return result


def calculate_events(states: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "r56_cfoa_slope8": "cfoa_slope8",
        "r56_revenue_assets_slope8": "revenue_assets_slope8",
        "r56_cash_support_slope8": "cash_support_slope8",
        "r56_nonprofit_trend_equal3": "nonprofit_trend_equal3",
        "r57_cash_earnings_yoy_gap_mean4": "cash_earnings_yoy_gap_mean4",
        "r57_cash_earnings_gap_convergence": "cash_earnings_gap_convergence",
        "r57_cash_earnings_state_equal2": "cash_earnings_state_equal2",
        "r58_financing_headroom_state": "headroom",
        "r58_financing_headroom_change4": "headroom_change4",
        "r58_constraint_transition_equal3": "constraint_transition_equal3",
        "r59_industry_financial_comparability4": "industry_financial_comparability4",
        "r59_peer_cashflow_diffusion_gap": "peer_cashflow_diffusion_gap",
    }
    pieces = []
    for factor, source in mapping.items():
        event = states[["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME"]].copy()
        event["factor"] = factor
        event["value"] = pd.to_numeric(states[source], errors="coerce")
        pieces.append(event.dropna(subset=["EVENT_TIME", "value"]))
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["factor", "SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"]
    )


def generate_sparse(panel: Path, pit_dir: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    events = calculate_events(build_quarterly_states(pit_dir))
    workflow.CANDIDATE_COLUMNS = CANDIDATE_COLUMNS
    wide = workflow.prepare_wide_events(events, calendar)
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
                   ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
        keys = _normalize_panel(pd.read_parquet(panel, columns=KEYS, filters=filters))
        mapped = workflow._map_sparse(keys, wide)
        chunks.append(mapped)
        for factor in CANDIDATE_COLUMNS:
            values = mapped[factor]
            coverage.append({"year": year, "factor": factor, "rows": len(mapped),
                             "observed_rows": int(values.notna().sum()),
                             "missing_rate_before_fill": float(values.isna().mean()),
                             "observed_days": int(mapped.loc[values.notna(), "TRADE_DATE"].nunique())})
        print(f"{year}: {len(mapped):,} rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    result.to_parquet(output_dir / "round56_59_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(output_dir / "round56_59_sparse_coverage.csv", index=False, encoding="utf-8-sig")
    (output_dir / "round56_59_metadata.json").write_text(json.dumps({
        "stage": "sparse_before_fill", "factors": CANDIDATE_COLUMNS,
        "uses_rank": False, "label_fitted": False, "q1_or_60d_restriction": False,
        "history_parameters": {"trend_quarters": 8, "trend_min_quarters": 6,
                               "state_change_quarters": 4},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def fill_after_test(sparse: Path, sparse_ic: Path, output: Path) -> None:
    tested = set(pd.read_csv(sparse_ic)["factor"].astype(str))
    missing_tests = sorted(set(CANDIDATE_COLUMNS) - tested)
    if missing_tests:
        raise RuntimeError(f"factors must be sparse-tested before filling: {missing_tests}")
    data = pd.read_parquet(sparse, columns=KEYS + CANDIDATE_COLUMNS)
    before = data[CANDIDATE_COLUMNS].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[CANDIDATE_COLUMNS].transform("median")
    data[CANDIDATE_COLUMNS] = data[CANDIDATE_COLUMNS].fillna(medians)
    whole_day = data[CANDIDATE_COLUMNS].isna().sum()
    data[CANDIDATE_COLUMNS] = data[CANDIDATE_COLUMNS].fillna(0.0)
    remaining = data[CANDIDATE_COLUMNS].isna().sum()
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False, compression="zstd")
    pd.DataFrame({
        "factor": CANDIDATE_COLUMNS,
        "missing_rate_before_fill": [float(before[f]) for f in CANDIDATE_COLUMNS],
        "fill_method": "same_date_median_then_zero_only_for_whole_unavailable_day_after_sparse_test",
        "whole_day_neutral_rows": [int(whole_day[f]) for f in CANDIDATE_COLUMNS],
        "remaining_missing_rows": [int(remaining[f]) for f in CANDIDATE_COLUMNS],
    }).to_csv(output.with_suffix(".fill_report.csv"), index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, required=True)
    generate.add_argument("--pit-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, required=True)
    fill.add_argument("--sparse-ic", type=Path, required=True)
    fill.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-sparse":
        generate_sparse(args.panel.resolve(), args.pit_dir.resolve(), args.output_dir.resolve())
    else:
        fill_after_test(args.sparse.resolve(), args.sparse_ic.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
