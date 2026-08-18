"""Build four second-priority PIT-safe composite financial factors.

Final columns
-------------
``profitability_quality_score``
    Mean daily percentile rank of operating-profit growth, CFO/assets,
    accrual quality, gross-margin change and asset-turnover change.

``management_mispricing_score``
    Mean daily percentile rank of accrual quality, low net operating assets,
    low asset growth and low investment intensity.

``accrual_nonrecurring_score``
    Mean daily percentile rank of accrual quality and low non-recurring
    profit/loss.

``fundamental_momentum``
    Mean daily percentile rank of revenue growth, operating-profit growth,
    deducted-profit growth, CFO surprise, gross-margin change and
    asset-turnover change.

All source metrics use first-valid PIT disclosures. Cumulative statement
values are converted to standalone quarters and strict four-quarter TTM
values. Final scores are formed only after source metrics are carried to the
daily panel, so ranks compare the full contemporaneously available cross
section rather than only firms announcing on the same day.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fundamental_priority_factors import (
    WINSOR_LIMITS,
    _carry_one_factor,
    _latest_time,
    _normalize_panel,
    _winsorize_daily,
    build_ttm_metric,
)
from fundamental_priority_factors_part2 import (
    build_balance_snapshot,
)
from quarterly_f_score import (
    COMMON_COLUMNS,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PIT_DIR = BASE_DIR / "data" / "new_pit"
DEFAULT_INDICATOR_PATH = BASE_DIR / "data" / "ch_models" / "earnings_pit"
DEFAULT_FACTOR_PATH = BASE_DIR / "factors.parquet"
DEFAULT_AUDIT_DIR = BASE_DIR / "factor_components"

KEYS = ["TRADE_DATE", "SECURITY_ID"]
EVENT_KEYS = ["SECURITY_ID", "QUARTER_INDEX"]
FACTOR_COLUMNS = [
    "profitability_quality_score",
    "management_mispricing_score",
    "accrual_nonrecurring_score",
    "fundamental_momentum",
]
EXISTING_COMPONENTS = [
    "operating_profit_growth",
    "cfo_sue",
    "accrual_quality",
    "asset_growth",
    "investment_to_assets",
]
NEW_COMPONENTS = [
    "cfo_to_assets",
    "gross_margin_change",
    "asset_turnover_change",
    "noa_quality",
    "nonrecurring_quality",
    "revenue_growth",
    "deducted_profit_growth",
]
INCOME_ITEMS = ["REVENUE", "COGS"]
CASHFLOW_ITEMS = ["N_CF_OPERATE_A"]
INDICATOR_ITEMS = ["N_INCOME_CUT", "NR_PROFIT_LOSS"]
NOA_BALANCE_ITEMS = [
    "T_ASSETS",
    "CASH_C_EQUIV",
    "T_LIAB",
    "ST_BORR",
    "NCL_WITHIN_1_Y",
    "LT_BORR",
    "BOND_PAYABLE",
]

SCORE_COMPONENTS = {
    "profitability_quality_score": [
        "operating_profit_growth",
        "cfo_to_assets",
        "accrual_quality",
        "gross_margin_change",
        "asset_turnover_change",
    ],
    "management_mispricing_score": [
        "accrual_quality",
        "noa_quality",
        "asset_growth",
        "investment_to_assets",
    ],
    "accrual_nonrecurring_score": [
        "accrual_quality",
        "nonrecurring_quality",
    ],
    "fundamental_momentum": [
        "revenue_growth",
        "operating_profit_growth",
        "deducted_profit_growth",
        "cfo_sue",
        "gross_margin_change",
        "asset_turnover_change",
    ],
}
MIN_COMPONENTS = {
    "profitability_quality_score": 4,
    "management_mispricing_score": 4,
    "accrual_nonrecurring_score": 2,
    "fundamental_momentum": 5,
}


def _read_inputs(
    pit_dir: Path,
    indicator_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    income = pd.read_parquet(
        pit_dir / "new_pit_income",
        columns=COMMON_COLUMNS + INCOME_ITEMS,
        engine="pyarrow",
    )
    cashflow = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=COMMON_COLUMNS + CASHFLOW_ITEMS,
        engine="pyarrow",
    )
    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=COMMON_COLUMNS
        + ["INDUSTRY_CATEGORY"]
        + NOA_BALANCE_ITEMS,
        engine="pyarrow",
    )
    indicator = pd.read_parquet(
        indicator_path,
        columns=[
            "ID",
            "SECURITY_ID",
            "ACT_PUBTIME",
            "END_DATE",
            "END_DATE_REP",
            "REPORT_TYPE",
            "FISCAL_PERIOD",
            "MERGED_FLAG",
        ]
        + INDICATOR_ITEMS,
        engine="pyarrow",
    )
    indicator["IS_CURRENT_PERIOD"] = True
    return income, cashflow, balance, indicator


def _build_ttm(
    statement: pd.DataFrame,
    column: str,
    name: str,
) -> pd.DataFrame:
    quarterly = build_standalone_quarterly_metric(
        statement,
        column,
        name=name,
    )
    return build_ttm_metric(quarterly, column)


def build_ttm_inputs(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    indicator: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "REVENUE": _build_ttm(income, "REVENUE", "利润表PIT"),
        "COGS": _build_ttm(income, "COGS", "利润表PIT"),
        "N_CF_OPERATE_A": _build_ttm(
            cashflow,
            "N_CF_OPERATE_A",
            "现金流量表PIT",
        ),
        "N_INCOME_CUT": _build_ttm(
            indicator,
            "N_INCOME_CUT",
            "财务指标PIT",
        ),
        "NR_PROFIT_LOSS": _build_ttm(
            indicator,
            "NR_PROFIT_LOSS",
            "财务指标PIT",
        ),
    }


def _shift_ttm(
    ttm: pd.DataFrame,
    column: str,
    quarters: int,
) -> pd.DataFrame:
    value = f"TTM_{column}"
    event = f"TTM_{column}_EVENT_TIME"
    shifted = ttm[EVENT_KEYS + [value, event]].copy()
    shifted["QUARTER_INDEX"] += quarters
    return shifted.rename(
        columns={
            value: f"LAG{quarters}_{value}",
            event: f"LAG{quarters}_{event}",
        }
    )


def _shift_balance(
    balance: pd.DataFrame,
    quarters: int,
    value_columns: list[str],
) -> pd.DataFrame:
    shifted = balance[
        EVENT_KEYS + ["BALANCE_EVENT_TIME"] + value_columns
    ].copy()
    shifted["QUARTER_INDEX"] += quarters
    return shifted.rename(
        columns={
            "BALANCE_EVENT_TIME": (
                f"LAG{quarters}_BALANCE_EVENT_TIME"
            ),
            **{
                column: f"LAG{quarters}_{column}"
                for column in value_columns
            },
        }
    )


def _select_event_columns(
    frame: pd.DataFrame,
    factor: str,
) -> pd.DataFrame:
    result = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[factor, "EVENT_TIME"]
    )
    return result[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            factor,
        ]
    ].copy()


def build_ttm_growth_events(
    ttm: pd.DataFrame,
    assets: pd.DataFrame,
    *,
    column: str,
    factor: str,
) -> pd.DataFrame:
    lag4_ttm = _shift_ttm(ttm, column, 4)
    lag4_assets = _shift_balance(assets, 4, ["T_ASSETS"])
    value = f"TTM_{column}"
    event = f"TTM_{column}_EVENT_TIME"
    base = pd.merge(
        ttm[
            EVENT_KEYS
            + [
                "FISCAL_YEAR",
                "FISCAL_QUARTER",
                "END_DATE",
                value,
                event,
            ]
        ],
        lag4_ttm,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    base = pd.merge(
        base,
        assets[EVENT_KEYS + ["BALANCE_EVENT_TIME"]],
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    base = pd.merge(
        base,
        lag4_assets,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    base[factor] = (
        (base[value] - base[f"LAG4_{value}"])
        .div(base["LAG4_T_ASSETS"])
        .where(base["LAG4_T_ASSETS"].gt(0))
    )
    base["EVENT_TIME"] = _latest_time(
        base,
        [
            event,
            f"LAG4_{event}",
            "BALANCE_EVENT_TIME",
            "LAG4_BALANCE_EVENT_TIME",
        ],
    )
    return _select_event_columns(base, factor)


def build_cfo_to_assets_events(
    cfo_ttm: pd.DataFrame,
    assets: pd.DataFrame,
) -> pd.DataFrame:
    lag4_assets = _shift_balance(assets, 4, ["T_ASSETS"])
    base = pd.merge(
        cfo_ttm[
            EVENT_KEYS
            + [
                "FISCAL_YEAR",
                "FISCAL_QUARTER",
                "END_DATE",
                "TTM_N_CF_OPERATE_A",
                "TTM_N_CF_OPERATE_A_EVENT_TIME",
            ]
        ],
        assets[
            EVENT_KEYS + ["T_ASSETS", "BALANCE_EVENT_TIME"]
        ],
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    base = pd.merge(
        base,
        lag4_assets,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    average_assets = (base["T_ASSETS"] + base["LAG4_T_ASSETS"]) / 2
    base["cfo_to_assets"] = base["TTM_N_CF_OPERATE_A"].div(
        average_assets
    ).where(average_assets.gt(0))
    base["EVENT_TIME"] = _latest_time(
        base,
        [
            "TTM_N_CF_OPERATE_A_EVENT_TIME",
            "BALANCE_EVENT_TIME",
            "LAG4_BALANCE_EVENT_TIME",
        ],
    )
    return _select_event_columns(base, "cfo_to_assets")


def build_margin_turnover_events(
    revenue_ttm: pd.DataFrame,
    cogs_ttm: pd.DataFrame,
    assets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lag4_revenue = _shift_ttm(revenue_ttm, "REVENUE", 4)
    lag4_cogs = _shift_ttm(cogs_ttm, "COGS", 4)
    lag4_assets = _shift_balance(assets, 4, ["T_ASSETS"])
    lag8_assets = _shift_balance(assets, 8, ["T_ASSETS"])

    base = revenue_ttm[
        EVENT_KEYS
        + [
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "TTM_REVENUE",
            "TTM_REVENUE_EVENT_TIME",
        ]
    ].copy()
    for other in [
        cogs_ttm[
            EVENT_KEYS + ["TTM_COGS", "TTM_COGS_EVENT_TIME"]
        ],
        lag4_revenue,
        lag4_cogs,
        assets[EVENT_KEYS + ["T_ASSETS", "BALANCE_EVENT_TIME"]],
        lag4_assets,
        lag8_assets,
    ]:
        base = pd.merge(
            base,
            other,
            on=EVENT_KEYS,
            how="inner",
            validate="one_to_one",
        )

    current_margin = (
        (base["TTM_REVENUE"] - base["TTM_COGS"])
        .div(base["TTM_REVENUE"])
        .where(base["TTM_REVENUE"].gt(0))
    )
    previous_margin = (
        (
            base["LAG4_TTM_REVENUE"]
            - base["LAG4_TTM_COGS"]
        )
        .div(base["LAG4_TTM_REVENUE"])
        .where(base["LAG4_TTM_REVENUE"].gt(0))
    )
    base["gross_margin_change"] = current_margin - previous_margin

    current_average_assets = (
        base["T_ASSETS"] + base["LAG4_T_ASSETS"]
    ) / 2
    previous_average_assets = (
        base["LAG4_T_ASSETS"] + base["LAG8_T_ASSETS"]
    ) / 2
    current_turnover = base["TTM_REVENUE"].div(
        current_average_assets
    ).where(current_average_assets.gt(0))
    previous_turnover = base["LAG4_TTM_REVENUE"].div(
        previous_average_assets
    ).where(previous_average_assets.gt(0))
    base["asset_turnover_change"] = current_turnover - previous_turnover
    base["EVENT_TIME"] = _latest_time(
        base,
        [
            "TTM_REVENUE_EVENT_TIME",
            "TTM_COGS_EVENT_TIME",
            "LAG4_TTM_REVENUE_EVENT_TIME",
            "LAG4_TTM_COGS_EVENT_TIME",
            "BALANCE_EVENT_TIME",
            "LAG4_BALANCE_EVENT_TIME",
            "LAG8_BALANCE_EVENT_TIME",
        ],
    )
    return (
        _select_event_columns(base, "gross_margin_change"),
        _select_event_columns(base, "asset_turnover_change"),
    )


def build_noa_quality_events(balance: pd.DataFrame) -> pd.DataFrame:
    prepared = balance.copy()
    debt_columns = [
        "ST_BORR",
        "NCL_WITHIN_1_Y",
        "LT_BORR",
        "BOND_PAYABLE",
    ]
    for column in debt_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        ).fillna(0.0)
    snapshot = build_balance_snapshot(prepared, NOA_BALANCE_ITEMS)
    lag4_assets = _shift_balance(snapshot, 4, ["T_ASSETS"])
    base = pd.merge(
        snapshot,
        lag4_assets,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    operating_assets = base["T_ASSETS"] - base["CASH_C_EQUIV"]
    interest_bearing_debt = sum(base[column] for column in debt_columns)
    operating_liabilities = base["T_LIAB"] - interest_bearing_debt
    net_operating_assets = operating_assets - operating_liabilities
    base["noa_quality"] = (
        -net_operating_assets.div(base["LAG4_T_ASSETS"])
    ).where(base["LAG4_T_ASSETS"].gt(0))
    base["EVENT_TIME"] = _latest_time(
        base,
        ["BALANCE_EVENT_TIME", "LAG4_BALANCE_EVENT_TIME"],
    )
    return _select_event_columns(base, "noa_quality")


def build_nonrecurring_events(
    nr_ttm: pd.DataFrame,
    assets: pd.DataFrame,
) -> pd.DataFrame:
    lag4_assets = _shift_balance(assets, 4, ["T_ASSETS"])
    base = pd.merge(
        nr_ttm[
            EVENT_KEYS
            + [
                "FISCAL_YEAR",
                "FISCAL_QUARTER",
                "END_DATE",
                "TTM_NR_PROFIT_LOSS",
                "TTM_NR_PROFIT_LOSS_EVENT_TIME",
            ]
        ],
        assets[EVENT_KEYS + ["T_ASSETS", "BALANCE_EVENT_TIME"]],
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    base = pd.merge(
        base,
        lag4_assets,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    average_assets = (base["T_ASSETS"] + base["LAG4_T_ASSETS"]) / 2
    base["nonrecurring_quality"] = (
        -base["TTM_NR_PROFIT_LOSS"].div(average_assets)
    ).where(average_assets.gt(0))
    base["EVENT_TIME"] = _latest_time(
        base,
        [
            "TTM_NR_PROFIT_LOSS_EVENT_TIME",
            "BALANCE_EVENT_TIME",
            "LAG4_BALANCE_EVENT_TIME",
        ],
    )
    return _select_event_columns(base, "nonrecurring_quality")


def build_component_events(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    balance: pd.DataFrame,
    indicator: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ttm = build_ttm_inputs(income, cashflow, indicator)
    assets = build_balance_snapshot(balance, ["T_ASSETS"])
    margin, turnover = build_margin_turnover_events(
        ttm["REVENUE"],
        ttm["COGS"],
        assets,
    )
    return {
        "cfo_to_assets": build_cfo_to_assets_events(
            ttm["N_CF_OPERATE_A"],
            assets,
        ),
        "gross_margin_change": margin,
        "asset_turnover_change": turnover,
        "noa_quality": build_noa_quality_events(balance),
        "nonrecurring_quality": build_nonrecurring_events(
            ttm["NR_PROFIT_LOSS"],
            assets,
        ),
        "revenue_growth": build_ttm_growth_events(
            ttm["REVENUE"],
            assets,
            column="REVENUE",
            factor="revenue_growth",
        ),
        "deducted_profit_growth": build_ttm_growth_events(
            ttm["N_INCOME_CUT"],
            assets,
            column="N_INCOME_CUT",
            factor="deducted_profit_growth",
        ),
    }


def calculate_composite_scores(
    component_data: pd.DataFrame,
) -> pd.DataFrame:
    """Convert source metrics to daily percentile ranks and average them."""
    missing = sorted(
        set(KEYS + EXISTING_COMPONENTS + NEW_COMPONENTS).difference(
            component_data.columns
        )
    )
    if missing:
        raise KeyError(f"复合因子输入缺少字段: {missing}")
    data = component_data[KEYS + EXISTING_COMPONENTS + NEW_COMPONENTS].copy()
    all_components = list(
        dict.fromkeys(
            component
            for components in SCORE_COMPONENTS.values()
            for component in components
        )
    )
    for component in all_components:
        values = pd.to_numeric(data[component], errors="coerce")
        data[component] = (
            values.groupby(data["TRADE_DATE"], sort=False)
            .rank(method="average", pct=True)
            .astype("float32")
        )

    result = data[KEYS].copy()
    for factor, components in SCORE_COMPONENTS.items():
        count = data[components].notna().sum(axis=1)
        score = data[components].mean(axis=1, skipna=True)
        result[factor] = score.where(
            count.ge(MIN_COMPONENTS[factor])
        ).astype("float64")
    return result


def build_daily_scores(
    events: dict[str, pd.DataFrame],
    existing_factors: pd.DataFrame,
) -> pd.DataFrame:
    panel = _normalize_panel(existing_factors).sort_values(KEYS).reset_index(
        drop=True
    )
    existing = existing_factors[KEYS + EXISTING_COMPONENTS].copy()
    existing["TRADE_DATE"] = pd.to_datetime(
        existing["TRADE_DATE"],
        errors="coerce",
    ).dt.normalize()
    existing["SECURITY_ID"] = pd.to_numeric(
        existing["SECURITY_ID"],
        errors="raise",
    ).astype("int64")
    existing = existing.sort_values(KEYS).reset_index(drop=True)
    if not existing[KEYS].equals(panel[KEYS]):
        raise RuntimeError("现有因子组件与日频面板主键不一致")

    components = existing.copy()
    for component in NEW_COMPONENTS:
        one = _carry_one_factor(events[component], panel, component)
        one = _winsorize_daily(one, component, WINSOR_LIMITS)
        if not one[KEYS].equals(panel[KEYS]):
            raise RuntimeError(f"{component}映射后主键顺序发生变化")
        components[component] = one[component].to_numpy()
    return calculate_composite_scores(components)


def append_factors_atomically(
    factor_path: str | Path,
    factor_values: pd.DataFrame,
) -> tuple[Path, Path]:
    path = Path(factor_path).resolve()
    existing = pd.read_parquet(path)
    existing["TRADE_DATE"] = (
        pd.to_datetime(existing["TRADE_DATE"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    existing["SECURITY_ID"] = pd.to_numeric(
        existing["SECURITY_ID"],
        errors="raise",
    ).astype("int64")
    if existing.duplicated(KEYS).any():
        raise ValueError("原factors.parquet存在重复键")
    values = factor_values[KEYS + FACTOR_COLUMNS].copy()
    if values.duplicated(KEYS).any():
        raise ValueError("第二优先级复合因子存在重复键")
    updated = pd.merge(
        existing.drop(columns=FACTOR_COLUMNS, errors="ignore"),
        values,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(updated) != len(existing):
        raise RuntimeError("加入第二优先级因子后行数发生变化")

    backup_dir = path.parent / "输出与测试" / "因子备份"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (
        f"{path.stem}_before_secondary_priority_factors{path.suffix}"
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


def run_secondary_priority_factors(
    pit_dir: str | Path = DEFAULT_PIT_DIR,
    indicator_path: str | Path = DEFAULT_INDICATOR_PATH,
    factor_path: str | Path = DEFAULT_FACTOR_PATH,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict:
    pit_dir = Path(pit_dir).resolve()
    indicator_path = Path(indicator_path).resolve()
    factor_path = Path(factor_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("读取三张PIT报表和扣非财务指标PIT...")
    income, cashflow, balance, indicator = _read_inputs(
        pit_dir,
        indicator_path,
    )
    print(
        f"  income={len(income):,}, cashflow={len(cashflow):,}, "
        f"balance={len(balance):,}, indicator={len(indicator):,}"
    )
    events = build_component_events(
        income,
        cashflow,
        balance,
        indicator,
    )
    for component in NEW_COMPONENTS:
        print(
            f"  {component}: events={len(events[component]):,}, "
            f"stocks={events[component]['SECURITY_ID'].nunique():,}"
        )

    factor_input = pd.read_parquet(
        factor_path,
        columns=KEYS + EXISTING_COMPONENTS,
        engine="pyarrow",
    )
    daily = build_daily_scores(events, factor_input)
    output_path, backup_path = append_factors_atomically(
        factor_path,
        daily,
    )

    daily_path = audit_dir / "secondary_priority_factors_daily.parquet"
    diagnostics_path = (
        audit_dir / "secondary_priority_factors_diagnostics.json"
    )
    daily.to_parquet(
        daily_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    diagnostics: dict[str, object] = {
        "score_components": SCORE_COMPONENTS,
        "minimum_components": MIN_COMPONENTS,
        "income_rows": len(income),
        "cashflow_rows": len(cashflow),
        "balance_rows": len(balance),
        "indicator_rows": len(indicator),
        "daily_panel_rows": len(daily),
        "factor_path": str(output_path),
        "backup_path": str(backup_path),
        "daily_path": str(daily_path),
        "components": {},
        "factors": {},
        "event_paths": {},
    }
    for component in NEW_COMPONENTS:
        event_path = audit_dir / f"{component}_events.parquet"
        events[component].to_parquet(
            event_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        diagnostics["event_paths"][component] = str(event_path)
        diagnostics["components"][component] = {
            "event_rows": len(events[component]),
            "event_stocks": int(
                events[component]["SECURITY_ID"].nunique()
            ),
            "event_start": str(events[component]["EVENT_TIME"].min()),
            "event_end": str(events[component]["EVENT_TIME"].max()),
        }
    for factor in FACTOR_COLUMNS:
        valid = daily[factor].notna()
        diagnostics["factors"][factor] = {
            "daily_non_null": int(valid.sum()),
            "daily_coverage": float(valid.mean()),
            "factor_start": str(
                daily.loc[valid, "TRADE_DATE"].min()
            ),
            "factor_end": str(
                daily.loc[valid, "TRADE_DATE"].max()
            ),
        }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return {
        "factor_path": output_path,
        "backup_path": backup_path,
        "daily_path": daily_path,
        "diagnostics_path": diagnostics_path,
        "diagnostics": diagnostics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造四个严格PIT的第二优先级复合财务因子"
    )
    parser.add_argument("--pit-dir", type=Path, default=DEFAULT_PIT_DIR)
    parser.add_argument(
        "--indicator-path",
        type=Path,
        default=DEFAULT_INDICATOR_PATH,
    )
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_secondary_priority_factors(
        args.pit_dir,
        args.indicator_path,
        args.factors,
        args.audit_dir,
    )


if __name__ == "__main__":
    main()
