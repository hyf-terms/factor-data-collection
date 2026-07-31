"""Package tested factors into effective/potential and likely-invalid folders.

The script keeps the project root intact.  It writes compact factor Parquet
subsets, construction code, documentation, and lightweight IC artifacts to:

* 有效因子
* 无效因子

Classification uses the latest available Barra-neutralized daily Spearman IC:

* confirmed: abs(mean IC) >= 0.035
* potential: 0.030 <= abs(mean IC) < 0.035
* stable exception: abs(mean IC) >= 0.025 and both validation and holdout
  abs(IC) >= 0.020 with the same sign
* likely invalid: candidates outside the rules above or without a valid IC
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from factors_neus_only import summarize_ic


KEYS = ["TRADE_DATE", "SECURITY_ID"]
CONFIRMED_THRESHOLD = 0.035
POTENTIAL_THRESHOLD = 0.030
STABLE_FULL_THRESHOLD = 0.025
STABLE_PERIOD_THRESHOLD = 0.020

TEST_DIRS = [
    "factor_test_output",
    "factor_test_output_event_financial",
    "factor_test_output_literature_financial",
    "factor_test_output_mohanram_g_score",
    "factor_test_output_priority_financial",
    "factor_test_output_priority_financial_part2",
    "factor_test_output_quarterly_f_score",
    "factor_test_output_quarterly_indicators",
    "factor_test_output_raw_q1_minimal",
    "factor_test_output_secondary_priority",
    "factor_test_output_unreplicated_financial",
    "factor_test_output_updated_label_top_factors",
]

FACTOR_SOURCES = {
    "主因子": "factors.parquet",
    "事件候选": "factor_components/event_financial_candidates.parquet",
    "文献候选": "factor_components/literature_financial_candidates.parquet",
    "新增文献候选": "factor_components/unreplicated_financial_candidates.parquet",
    "单季度指标候选": "factor_components/quarterly_indicator_candidates.parquet",
    "原始少字段候选": "factor_components/raw_q1_minimal_candidates.parquet",
}

EFFECTIVE_CODE = [
    "organize_factor_packages.py",
    "event_financial_factor_search.py",
    "literature_financial_factor_search.py",
    "pead_sue_factor.py",
    "quarterly_f_score.py",
    "fundamental_priority_factors.py",
    "fundamental_priority_factors_part2.py",
    "secondary_priority_factors.py",
    "quarterly_indicator_factor_search.py",
    "raw_q1_minimal_factor_search.py",
    "gross_profitability_factor.py",
    "factors_neus_only.py",
]

INVALID_CODE = [
    "organize_factor_packages.py",
    "gross_profitability_factor.py",
    "pead_sue_factor.py",
    "quarterly_f_score.py",
    "mohanram_g_score.py",
    "fundamental_priority_factors.py",
    "fundamental_priority_factors_part2.py",
    "secondary_priority_factors.py",
    "quarterly_indicator_factor_search.py",
    "raw_q1_minimal_factor_search.py",
    "unreplicated_financial_factor_search.py",
    "factors_neus_only.py",
]

EFFECTIVE_DOCS = [
    "事件型财务因子说明.md",
    "文献财务因子扩展说明.md",
    "PEAD_SUE因子说明.md",
    "季度F-score因子说明.md",
    "第一批财务因子说明.md",
    "第二批财务因子说明.md",
    "第二优先级复合因子说明.md",
    "单季度指标财务因子挖掘说明.md",
    "原始报表少字段因子挖掘说明.md",
    "毛利盈利因子复现说明.md",
    "因子中性化与IC测试说明.md",
]

INVALID_DOCS = [
    "毛利盈利因子复现说明.md",
    "PEAD_SUE因子说明.md",
    "季度F-score因子说明.md",
    "Mohanram_G-score因子说明.md",
    "第一批财务因子说明.md",
    "第二批财务因子说明.md",
    "第二优先级复合因子说明.md",
    "单季度指标财务因子挖掘说明.md",
    "原始报表少字段因子挖掘说明.md",
    "新增文献财务因子说明.md",
    "因子中性化与IC测试说明.md",
]

LIGHT_TEST_FILES = [
    "daily_ic.parquet",
    "ic_summary.csv",
    "ic_summary.parquet",
    "merge_diagnostics.parquet",
    "robustness_by_period.csv",
    "robustness_by_period.parquet",
    "run_metadata.json",
]


def _period_statistics(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["TRADE_DATE"] = pd.to_datetime(daily["TRADE_DATE"])
    periods = {
        "train_2017_2022": ("2017-01-01", "2022-12-31"),
        "validation_2023_2024": ("2023-01-01", "2024-12-31"),
        "holdout_2025_2026": ("2025-01-01", "2026-12-31"),
    }
    rows: list[dict[str, object]] = []
    for factor, group in daily.groupby("factor", sort=False):
        for period, (start, end) in periods.items():
            sample = group.loc[group["TRADE_DATE"].between(start, end)]
            valid = sample.dropna(subset=["neutral_ic"])
            rows.append(
                {
                    "factor": factor,
                    "period": period,
                    "neutral_mean_ic": valid["neutral_ic"].mean(),
                    "effective_days": len(valid),
                    "average_cross_section": valid["neutral_n"].mean(),
                }
            )
    return pd.DataFrame(rows)


def refresh_robustness_files(base: Path) -> None:
    """Refresh period tables so no old-label robustness file remains."""
    for name in [
        "factor_test_output_event_financial",
        "factor_test_output_literature_financial",
        "factor_test_output_quarterly_indicators",
        "factor_test_output_raw_q1_minimal",
        "factor_test_output_unreplicated_financial",
        "factor_test_output_updated_label_top_factors",
    ]:
        directory = base / name
        daily_path = directory / "daily_ic.parquet"
        if not daily_path.exists():
            continue
        result = _period_statistics(pd.read_parquet(daily_path))
        result.to_csv(
            directory / "robustness_by_period.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result.to_parquet(
            directory / "robustness_by_period.parquet",
            index=False,
        )


def align_all_test_results_to_current_label(base: Path) -> pd.Timestamp:
    """Drop obsolete tail dates and refresh every existing IC summary.

    Factor residuals do not depend on the label and are intentionally left
    untouched.  The daily IC and summary files are rewritten to the maximum
    factor date available in the current label file.
    """
    label_dates = pd.read_parquet(
        base / "label.parquet",
        columns=["TRADE_DATE"],
    )
    label_end = pd.to_datetime(label_dates["TRADE_DATE"]).max()
    if pd.isna(label_end):
        raise ValueError("label.parquet has no valid TRADE_DATE")

    for name in TEST_DIRS:
        directory = base / name
        daily_path = directory / "daily_ic.parquet"
        if not daily_path.exists():
            continue
        daily = pd.read_parquet(daily_path)
        daily["TRADE_DATE"] = pd.to_datetime(daily["TRADE_DATE"])
        daily = daily.loc[daily["TRADE_DATE"].le(label_end)].copy()
        daily.to_parquet(daily_path, index=False)

        summary = summarize_ic(daily)
        summary.to_csv(
            directory / "ic_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_parquet(
            directory / "ic_summary.parquet",
            index=False,
        )

        metadata_path = directory / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["effective_label_trade_date_end"] = str(
                label_end.date()
            )
            metadata["daily_ic_aligned_to_current_label"] = True
            metadata["alignment_note"] = (
                "Daily IC was restricted to the current label date range; "
                "factor residuals are label-independent and were retained."
            )
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return label_end


def collect_latest_results(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect one latest neutral result and its period statistics per factor."""
    summary_pieces: list[pd.DataFrame] = []
    period_pieces: list[pd.DataFrame] = []
    for name in TEST_DIRS:
        directory = base / name
        summary_path = directory / "ic_summary.parquet"
        daily_path = directory / "daily_ic.parquet"
        if not summary_path.exists():
            continue
        summary = pd.read_parquet(summary_path)
        summary = summary.loc[summary["version"].eq("neutral")].copy()
        summary["test_directory"] = name
        summary["result_mtime_ns"] = summary_path.stat().st_mtime_ns
        summary_pieces.append(summary)
        if daily_path.exists():
            periods = _period_statistics(pd.read_parquet(daily_path))
            periods["test_directory"] = name
            periods["result_mtime_ns"] = summary_path.stat().st_mtime_ns
            period_pieces.append(periods)

    summaries = pd.concat(summary_pieces, ignore_index=True)
    summaries = (
        summaries.sort_values("result_mtime_ns")
        .drop_duplicates("factor", keep="last")
        .reset_index(drop=True)
    )
    periods = pd.concat(period_pieces, ignore_index=True)
    latest_keys = summaries[
        ["factor", "test_directory", "result_mtime_ns"]
    ]
    periods = periods.merge(
        latest_keys,
        on=["factor", "test_directory", "result_mtime_ns"],
        how="inner",
        validate="many_to_one",
    )
    return summaries, periods


def build_catalog(
    base: Path,
    summaries: pd.DataFrame,
    periods: pd.DataFrame,
) -> pd.DataFrame:
    source_lookup: dict[str, list[str]] = {}
    for source, relative in FACTOR_SOURCES.items():
        path = base / relative
        if not path.exists():
            continue
        for column in pq.read_schema(path).names:
            if column not in KEYS:
                source_lookup.setdefault(column, []).append(source)

    catalog = pd.DataFrame(
        {
            "factor": sorted(source_lookup),
            "data_source": [
                "、".join(source_lookup[factor])
                for factor in sorted(source_lookup)
            ],
        }
    )
    keep = [
        "factor",
        "mean_ic",
        "abs_mean_ic",
        "std_ic",
        "icir",
        "t_stat",
        "sign_win_rate",
        "effective_days",
        "average_cross_section",
        "test_directory",
    ]
    catalog = catalog.merge(
        summaries[keep],
        on="factor",
        how="left",
        validate="one_to_one",
    )
    period_wide = periods.pivot(
        index="factor",
        columns="period",
        values="neutral_mean_ic",
    ).reset_index()
    catalog = catalog.merge(
        period_wide,
        on="factor",
        how="left",
        validate="one_to_one",
    )

    def classify(row: pd.Series) -> tuple[str, str]:
        value = row["abs_mean_ic"]
        if pd.isna(value):
            return "大概率无效（无有效IC）", "没有有效中性化IC"
        if value >= CONFIRMED_THRESHOLD:
            return "已满足条件", f"|IC| >= {CONFIRMED_THRESHOLD:.3f}"
        if value >= POTENTIAL_THRESHOLD:
            return (
                "潜在有效",
                f"{POTENTIAL_THRESHOLD:.3f} <= |IC| < "
                f"{CONFIRMED_THRESHOLD:.3f}",
            )
        validation = row.get("validation_2023_2024")
        holdout = row.get("holdout_2025_2026")
        mean_ic = row.get("mean_ic")
        stable = (
            value >= STABLE_FULL_THRESHOLD
            and pd.notna(validation)
            and pd.notna(holdout)
            and abs(validation) >= STABLE_PERIOD_THRESHOLD
            and abs(holdout) >= STABLE_PERIOD_THRESHOLD
            and mean_ic * validation > 0
            and mean_ic * holdout > 0
        )
        if stable:
            return "潜在有效", "全期、验证期和留出期方向一致且达到稳定性门槛"
        return "大概率无效", "未达到潜在有效门槛"

    classified = catalog.apply(classify, axis=1, result_type="expand")
    classified.columns = ["classification", "classification_reason"]
    catalog[classified.columns] = classified
    priority = {
        "已满足条件": 0,
        "潜在有效": 1,
        "大概率无效": 2,
        "大概率无效（无有效IC）": 3,
    }
    catalog["classification_order"] = catalog["classification"].map(priority)
    return (
        catalog.sort_values(
            ["classification_order", "abs_mean_ic", "factor"],
            ascending=[True, False, True],
            na_position="last",
        )
        .drop(columns="classification_order")
        .reset_index(drop=True)
    )


def _write_factor_subset(
    source: Path,
    target: Path,
    factor_columns: list[str],
) -> None:
    if not factor_columns:
        return
    table = pq.read_table(source, columns=KEYS + factor_columns)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target, compression="zstd")


def write_factor_data(
    base: Path,
    catalog: pd.DataFrame,
    effective_dir: Path,
    invalid_dir: Path,
) -> None:
    classification = catalog.set_index("factor")["classification"].to_dict()
    for source_name, relative in FACTOR_SOURCES.items():
        source = base / relative
        if not source.exists():
            continue
        columns = [
            column
            for column in pq.read_schema(source).names
            if column not in KEYS
        ]
        effective = [
            column
            for column in columns
            if classification.get(column, "").startswith(("已", "潜在"))
        ]
        invalid = [
            column for column in columns if column not in effective
        ]
        safe_name = source_name.replace("/", "_")
        _write_factor_subset(
            source,
            effective_dir / "因子数据" / f"{safe_name}_有效及潜在.parquet",
            effective,
        )
        _write_factor_subset(
            source,
            invalid_dir / "因子数据" / f"{safe_name}_大概率无效.parquet",
            invalid,
        )


def _copy_named_files(base: Path, names: list[str], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = base / name
        if source.exists():
            shutil.copy2(source, target / source.name)


def copy_light_test_results(
    base: Path,
    catalog: pd.DataFrame,
    effective_dir: Path,
    invalid_dir: Path,
) -> None:
    effective_factors = set(
        catalog.loc[
            catalog["classification"].str.startswith(("已", "潜在")),
            "factor",
        ]
    )
    invalid_factors = set(catalog["factor"]) - effective_factors
    for name in TEST_DIRS:
        source_dir = base / name
        summary_path = source_dir / "ic_summary.parquet"
        if not summary_path.exists():
            continue
        summary = pd.read_parquet(summary_path)
        factors = set(summary["factor"])
        destinations = [
            (
                effective_dir / "测试结果" / name,
                factors & effective_factors,
            ),
            (
                invalid_dir / "测试结果" / name,
                factors & invalid_factors,
            ),
        ]
        for destination, selected in destinations:
            if not selected:
                continue
            destination.mkdir(parents=True, exist_ok=True)
            filtered_summary = summary.loc[
                summary["factor"].isin(selected)
            ].copy()
            filtered_summary.to_csv(
                destination / "ic_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            filtered_summary.to_parquet(
                destination / "ic_summary.parquet",
                index=False,
            )

            daily_path = source_dir / "daily_ic.parquet"
            if daily_path.exists():
                daily = pd.read_parquet(daily_path)
                daily = daily.loc[daily["factor"].isin(selected)].copy()
                daily.to_parquet(
                    destination / "daily_ic.parquet",
                    index=False,
                )
                robustness = _period_statistics(daily)
                robustness.to_csv(
                    destination / "robustness_by_period.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                robustness.to_parquet(
                    destination / "robustness_by_period.parquet",
                    index=False,
                )

            # Remove general files copied by an older packaging run.  They
            # describe the unfiltered source test and would be misleading in
            # a category-specific package.
            for filename in [
                "merge_diagnostics.parquet",
                "run_metadata.json",
            ]:
                stale = destination / filename
                if stale.exists():
                    stale.unlink()


def write_readmes(
    catalog: pd.DataFrame,
    effective_dir: Path,
    invalid_dir: Path,
) -> None:
    confirmed = catalog.loc[catalog["classification"].eq("已满足条件")]
    potential = catalog.loc[catalog["classification"].eq("潜在有效")]
    invalid = catalog.loc[
        catalog["classification"].str.startswith("大概率无效")
    ]

    effective_text = f"""# 有效因子包

本目录按最新 `label.parquet` 和每日 Barra 残差化 Spearman IC 整理。

- 已满足条件：`|中性化平均IC| >= {CONFIRMED_THRESHOLD:.3f}`，共 {len(confirmed)} 个；
- 潜在有效：`{POTENTIAL_THRESHOLD:.3f} <= |IC| < {CONFIRMED_THRESHOLD:.3f}`，共 {len(potential)} 个；
- 稳定性例外：全期 `|IC| >= {STABLE_FULL_THRESHOLD:.3f}`，且验证期和留出期
  `|IC| >= {STABLE_PERIOD_THRESHOLD:.3f}`、方向一致，也归入潜在有效；
- 因子构造严格使用 PIT 披露时间，候选生成不读取标签。

`因子分类清单.csv` 是最终判断依据；`因子数据` 保存按来源拆分的
Parquet；`测试结果` 不复制体积很大的中性化残差文件，完整残差仍保留在项目
根目录原测试文件夹中。

注意：潜在有效不代表已经通过样本外验证，仍应观察验证期、留出期、覆盖范围和
交易成本。
"""
    invalid_text = f"""# 大概率无效因子包

本目录收纳没有达到确认、潜在或分期稳定门槛的候选。这是当前标签、当前股票池
和当前Barra口径下的经验分类，不代表这些财务指标在其他持有期、事件窗口或
组合模型中永久无效。

`因子分类清单.csv` 记录全期及分期IC。原始构造程序保留，便于未来改变标签、
窗口或中性化方法后重新检验。
"""
    (effective_dir / "README.md").write_text(effective_text, encoding="utf-8")
    (invalid_dir / "README.md").write_text(invalid_text, encoding="utf-8")

    catalog.loc[
        catalog["classification"].str.startswith(("已", "潜在"))
    ].to_csv(
        effective_dir / "因子分类清单.csv",
        index=False,
        encoding="utf-8-sig",
    )
    catalog.loc[
        catalog["classification"].str.startswith("大概率无效")
    ].to_csv(
        invalid_dir / "因子分类清单.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "confirmed_threshold": CONFIRMED_THRESHOLD,
        "potential_threshold": POTENTIAL_THRESHOLD,
        "stable_full_threshold": STABLE_FULL_THRESHOLD,
        "stable_period_threshold": STABLE_PERIOD_THRESHOLD,
        "confirmed_count": len(confirmed),
        "potential_count": len(potential),
        "likely_invalid_count": len(invalid),
    }
    for directory in (effective_dir, invalid_dir):
        (directory / "分类元数据.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def organize(base: Path) -> None:
    effective_dir = base / "有效因子"
    invalid_dir = base / "无效因子"
    effective_dir.mkdir(exist_ok=True)
    invalid_dir.mkdir(exist_ok=True)

    label_end = align_all_test_results_to_current_label(base)
    refresh_robustness_files(base)
    summaries, periods = collect_latest_results(base)
    catalog = build_catalog(base, summaries, periods)
    catalog.to_csv(
        base / "因子分类总表.csv",
        index=False,
        encoding="utf-8-sig",
    )
    catalog.to_parquet(base / "因子分类总表.parquet", index=False)

    write_factor_data(base, catalog, effective_dir, invalid_dir)
    _copy_named_files(base, EFFECTIVE_CODE, effective_dir / "构造程序")
    _copy_named_files(base, INVALID_CODE, invalid_dir / "构造程序")
    _copy_named_files(base, EFFECTIVE_DOCS, effective_dir / "说明文档")
    _copy_named_files(base, INVALID_DOCS, invalid_dir / "说明文档")
    copy_light_test_results(base, catalog, effective_dir, invalid_dir)
    write_readmes(catalog, effective_dir, invalid_dir)

    print(
        catalog.groupby("classification", dropna=False)
        .size()
        .sort_index()
        .to_string()
    )
    print(f"Wrote: {effective_dir}")
    print(f"Wrote: {invalid_dir}")
    print(f"Aligned daily IC through: {label_end.date()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    organize(args.base_dir.resolve())
