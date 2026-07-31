"""Barra-neutralize factor data and evaluate daily cross-sectional IC.

Default input files are located beside this script:

* factors.parquet
* barra_diy.parquet
* label.parquet

The implementation processes one calendar year at a time so that a large
Barra panel does not need to be loaded into memory all at once.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
KEYS = ["TRADE_DATE", "SECURITY_ID"]
FACTOR_PATH = BASE_DIR / "factors.parquet"
BARRA_PATH = BASE_DIR / "barra_diy.parquet"
LABEL_PATH = BASE_DIR / "label.parquet"
OUTPUT_DIR = BASE_DIR / "factor_test_output"

STYLE_FACTORS = [
    "liquidity",
    "leverage",
    "earnings_variability",
    "earnings_quality",
    "profitability",
    "investment_quality",
    "book_to_price",
    "earnings_yield",
    "longterm_reversal",
    "growth",
    "momentum",
    "mid_cap",
    "size",
    "beta",
    "residual_volatility",
    "dividend_yield",
    "industry_momentum",
    "sentiment",
    "seasonality",
    "shortterm_reversal",
]

# "未分类" is deliberately omitted and acts as the reference industry.
INDUSTRY_FACTORS = [
    "交通运输",
    "传媒",
    "农林牧渔",
    "医药",
    "商贸零售",
    "国防军工",
    "基础化工",
    "家电",
    "建材",
    "建筑",
    "房地产",
    "有色金属",
    "机械",
    "汽车",
    "消费者服务",
    "煤炭",
    "电力及公用事业",
    "电力设备及新能源",
    "电子",
    "石油石化",
    "纺织服装",
    "综合",
    "综合金融",
    "计算机",
    "轻工制造",
    "通信",
    "钢铁",
    "银行",
    "非银行金融",
    "食品饮料",
]

BARRA_FACTORS = STYLE_FACTORS + INDUSTRY_FACTORS
NON_FACTOR_COLUMNS = {
    *KEYS,
    "LABEL",
    "LABEL_LATE5",
    "LABELD",
    "RET",
    "__index_level_0__",
}


@dataclass
class YearResult:
    residualized: pd.DataFrame
    daily_ic: pd.DataFrame
    diagnostics: dict
    invalid_factors: list[str]
    max_abs_moment: float


def restore_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore keys stored in a Parquet index and normalize their types."""
    data = frame.copy()
    if any(column not in data.columns for column in KEYS):
        data = data.reset_index()
    missing = sorted(set(KEYS).difference(data.columns))
    if missing:
        raise KeyError(f"数据缺少键列: {missing}")
    data["TRADE_DATE"] = (
        pd.to_datetime(data["TRADE_DATE"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    data["SECURITY_ID"] = pd.to_numeric(
        data["SECURITY_ID"],
        errors="coerce",
    )
    data = data.dropna(subset=KEYS).copy()
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    return data


def _validate_unique_keys(frame: pd.DataFrame, name: str) -> None:
    duplicate = frame.duplicated(KEYS, keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, KEYS].head(10).to_dict("records")
        raise ValueError(
            f"{name} 存在 {int(duplicate.sum()):,} 行重复键，示例: {examples}"
        )


def _schema_names(path: str | Path) -> list[str]:
    return pq.read_schema(Path(path)).names


def infer_factor_columns(
    factor_path: str | Path,
    requested: Iterable[str] | None = None,
) -> list[str]:
    """Return numeric factor columns, excluding keys and known label fields."""
    names = _schema_names(factor_path)
    candidates = (
        list(requested)
        if requested
        else [name for name in names if name not in NON_FACTOR_COLUMNS]
    )
    missing = sorted(set(candidates).difference(names))
    if missing:
        raise KeyError(f"factors.parquet 缺少指定因子列: {missing}")
    if not candidates:
        raise ValueError("factors.parquet 中没有可测试的因子列")
    return candidates


def _date_range(path: str | Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    parquet = pq.ParquetFile(Path(path))
    column_index = parquet.schema_arrow.get_field_index("TRADE_DATE")
    if column_index < 0:
        raise KeyError(f"{path} 缺少 TRADE_DATE")
    minima: list[pd.Timestamp] = []
    maxima: list[pd.Timestamp] = []
    for index in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(index).column(
            column_index
        ).statistics
        if statistics and statistics.has_min_max:
            minima.append(pd.Timestamp(statistics.min).normalize())
            maxima.append(pd.Timestamp(statistics.max).normalize())
    if not minima:
        dates = pd.read_parquet(path, columns=["TRADE_DATE"])["TRADE_DATE"]
        return pd.Timestamp(dates.min()).normalize(), pd.Timestamp(
            dates.max()
        ).normalize()
    return min(minima), max(maxima)


def _read_date_slice(
    path: str | Path,
    columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    data = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("TRADE_DATE", ">=", start),
            ("TRADE_DATE", "<=", end),
        ],
        engine="pyarrow",
    )
    return restore_keys(data)


def residualize_values_by_date(
    frame: pd.DataFrame,
    value_cols: list[str],
    exposure_cols: list[str],
) -> tuple[pd.DataFrame, float]:
    """Run daily OLS with an intercept and replace values by residuals."""
    out = frame.copy()
    # Candidate files may use float32 to reduce disk usage. OLS produces
    # float64 residuals, so promote explicitly before assignment.
    out[value_cols] = out[value_cols].astype("float64")
    max_abs_moment = 0.0
    for _, indices in frame.groupby("TRADE_DATE", sort=True).groups.items():
        group = frame.loc[indices]
        design = np.column_stack(
            [
                np.ones(len(group), dtype=np.float64),
                group[exposure_cols].to_numpy(dtype=np.float64),
            ]
        )
        values = group[value_cols].to_numpy(dtype=np.float64)
        residual = _ols_residual(design, values)
        out.loc[indices, value_cols] = residual
        moment = design.T @ residual / max(len(group), 1)
        max_abs_moment = max(
            max_abs_moment,
            float(np.max(np.abs(moment))),
        )
    return out, max_abs_moment


def _ols_residual(
    design: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Return OLS residuals, using fast QR when the design is full-rank."""
    varying = np.ptp(design, axis=0) > 1e-12
    varying[0] = True
    reduced = design[:, varying]
    q_matrix, r_matrix = np.linalg.qr(reduced, mode="reduced")
    diagonal = np.abs(np.diag(r_matrix))
    tolerance = (
        max(reduced.shape)
        * np.finfo(np.float64).eps
        * (float(diagonal.max()) if diagonal.size else 0.0)
    )
    if diagonal.size and np.all(diagonal > tolerance):
        return values - q_matrix @ (q_matrix.T @ values)
    coefficients, *_ = np.linalg.lstsq(reduced, values, rcond=None)
    return values - reduced @ coefficients


def _residualize_one_factor(
    base: pd.DataFrame,
    factor: str,
    exposure_cols: list[str],
    min_cross_section: int,
) -> tuple[pd.DataFrame, float]:
    columns = KEYS + [factor] + exposure_cols
    data = base[columns].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[factor] + exposure_cols).copy()
    required_size = max(
        min_cross_section,
        len(exposure_cols) + 5,
    )
    daily_size = data.groupby("TRADE_DATE", sort=False).size()
    valid_dates = daily_size[daily_size >= required_size].index
    data = data.loc[data["TRADE_DATE"].isin(valid_dates)].copy()
    if data.empty:
        return data[KEYS + [factor]], 0.0
    residualized, moment = residualize_values_by_date(
        data,
        [factor],
        exposure_cols,
    )
    return residualized[KEYS + [factor]], moment


def residualize_factor_matrix_by_date(
    base: pd.DataFrame,
    factor_cols: list[str],
    exposure_cols: list[str],
    min_cross_section: int,
) -> tuple[pd.DataFrame, list[str], float]:
    """Residualize factors in batches that share the same valid-row mask.

    ``np.linalg.lstsq`` accepts several right-hand sides.  Financial event
    factors commonly have the same 60-day availability mask, so this avoids
    decomposing an identical daily Barra design matrix once per factor.
    """
    required_size = max(min_cross_section, len(exposure_cols) + 5)
    values_out = np.full(
        (len(base), len(factor_cols)),
        np.nan,
        dtype=np.float64,
    )
    factor_positions = {
        factor: position
        for position, factor in enumerate(factor_cols)
    }
    max_abs_moment = 0.0

    for positions in base.groupby(
        "TRADE_DATE", sort=True
    ).indices.values():
        group = base.iloc[positions]
        exposure_values = group[exposure_cols].to_numpy(
            dtype=np.float64
        )
        exposure_valid = np.isfinite(exposure_values).all(axis=1)
        if int(exposure_valid.sum()) < required_size:
            continue
        factor_values = group[factor_cols].to_numpy(dtype=np.float64)
        mask_groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
        for column_position in range(len(factor_cols)):
            valid = exposure_valid & np.isfinite(
                factor_values[:, column_position]
            )
            if int(valid.sum()) < required_size:
                continue
            key = valid.tobytes()
            if key not in mask_groups:
                mask_groups[key] = (valid, [])
            mask_groups[key][1].append(column_position)

        for valid, columns in mask_groups.values():
            design = np.column_stack(
                [
                    np.ones(int(valid.sum()), dtype=np.float64),
                    exposure_values[valid],
                ]
            )
            targets = factor_values[np.ix_(valid, columns)]
            residual = _ols_residual(design, targets)
            row_positions = np.asarray(positions)[valid]
            values_out[np.ix_(row_positions, columns)] = residual
            moment = design.T @ residual / max(len(design), 1)
            max_abs_moment = max(
                max_abs_moment,
                float(np.max(np.abs(moment))),
            )

    residualized = base[KEYS].copy()
    for factor, position in factor_positions.items():
        residualized[factor] = values_out[:, position]
    invalid = [
        factor
        for factor in factor_cols
        if residualized[factor].notna().sum() == 0
    ]
    residualized = residualized.dropna(
        subset=factor_cols, how="all"
    ).reset_index(drop=True)
    return residualized, invalid, max_abs_moment


def _rank_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    x_rank = x.rank(method="average").to_numpy(dtype=np.float64)
    y_rank = y.rank(method="average").to_numpy(dtype=np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = np.sqrt(
        np.dot(x_rank, x_rank) * np.dot(y_rank, y_rank)
    )
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan
    return float(np.dot(x_rank, y_rank) / denominator)


def calculate_daily_spearman_ic(
    factor_data: pd.DataFrame,
    label_data: pd.DataFrame,
    factor: str,
    label_column: str = "LABEL",
    min_cross_section: int = 30,
) -> pd.DataFrame:
    """Calculate daily cross-sectional Spearman IC without SciPy."""
    merged = pd.merge(
        factor_data[KEYS + [factor]],
        label_data[KEYS + [label_column]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    return _calculate_daily_ic_from_merged(
        merged,
        factor,
        label_column,
        min_cross_section,
    )


def _calculate_daily_ic_from_merged(
    merged: pd.DataFrame,
    factor: str,
    label_column: str,
    min_cross_section: int,
) -> pd.DataFrame:
    """Calculate daily IC when keys, factor and label are already aligned."""
    merged = merged[KEYS + [factor, label_column]].replace(
        [np.inf, -np.inf], np.nan
    )
    merged = merged.dropna(subset=[factor, label_column])
    records: list[dict] = []
    for trade_date, group in merged.groupby("TRADE_DATE", sort=True):
        size = len(group)
        if (
            size < min_cross_section
            or group[factor].nunique(dropna=True) < 2
            or group[label_column].nunique(dropna=True) < 2
        ):
            correlation = np.nan
        else:
            correlation = _rank_correlation(
                group[factor],
                group[label_column],
            )
        records.append(
            {
                "TRADE_DATE": trade_date,
                "ic": correlation,
                "n": size,
            }
        )
    return pd.DataFrame(records, columns=["TRADE_DATE", "ic", "n"])


def _combine_ic_versions(
    factor: str,
    raw_ic: pd.DataFrame,
    matched_ic: pd.DataFrame,
    neutral_ic: pd.DataFrame,
) -> pd.DataFrame:
    def rename(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return frame.rename(
            columns={
                "ic": f"{prefix}_ic",
                "n": f"{prefix}_n",
            }
        )

    combined = pd.merge(
        rename(raw_ic, "raw"),
        rename(matched_ic, "raw_matched"),
        on="TRADE_DATE",
        how="outer",
    )
    combined = pd.merge(
        combined,
        rename(neutral_ic, "neutral"),
        on="TRADE_DATE",
        how="outer",
    )
    combined.insert(1, "factor", factor)
    return combined.sort_values("TRADE_DATE").reset_index(drop=True)


def _process_year(
    factor_path: Path,
    barra_path: Path,
    label_path: Path,
    factor_cols: list[str],
    exposure_cols: list[str],
    label_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_cross_section: int,
    min_ic_cross_section: int,
) -> YearResult:
    factors = _read_date_slice(
        factor_path,
        KEYS + factor_cols,
        start,
        end,
    )
    barra = _read_date_slice(
        barra_path,
        KEYS + exposure_cols,
        start,
        end,
    )
    labels = _read_date_slice(
        label_path,
        KEYS + [label_column],
        start,
        end,
    )
    _validate_unique_keys(factors, "factors")
    _validate_unique_keys(barra, "Barra")
    _validate_unique_keys(labels, "label")

    for column in factor_cols:
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    for column in exposure_cols:
        barra[column] = pd.to_numeric(barra[column], errors="coerce")
    labels[label_column] = pd.to_numeric(
        labels[label_column],
        errors="coerce",
    )

    factor_barra = pd.merge(
        factors,
        barra,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    factor_label = pd.merge(
        factors,
        labels,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    factor_label_rows = len(factor_label)

    daily_results: list[pd.DataFrame] = []
    residualized_frame, invalid_factors, max_abs_moment = (
        residualize_factor_matrix_by_date(
            factor_barra,
            factor_cols,
            exposure_cols,
            min_cross_section,
        )
    )
    evaluation = factor_label.merge(
        residualized_frame,
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("_RAW", "_NEUTRAL"),
    )

    for factor in factor_cols:
        residualized = residualized_frame[KEYS + [factor]].dropna(
            subset=[factor]
        )
        if residualized.empty:
            continue

        raw_column = f"{factor}_RAW"
        neutral_column = f"{factor}_NEUTRAL"
        raw_frame = evaluation[
            KEYS + [raw_column, label_column]
        ].rename(columns={raw_column: factor})
        neutral_frame = evaluation[
            KEYS + [neutral_column, label_column]
        ].rename(columns={neutral_column: factor})
        matched_raw_frame = raw_frame.loc[
            evaluation[neutral_column].notna()
        ]
        raw_ic = _calculate_daily_ic_from_merged(
            raw_frame,
            factor,
            label_column,
            min_ic_cross_section,
        )
        raw_matched_ic = _calculate_daily_ic_from_merged(
            matched_raw_frame,
            factor,
            label_column,
            min_ic_cross_section,
        )
        neutral_ic = _calculate_daily_ic_from_merged(
            neutral_frame,
            factor,
            label_column,
            min_ic_cross_section,
        )
        daily_results.append(
            _combine_ic_versions(
                factor,
                raw_ic,
                raw_matched_ic,
                neutral_ic,
            )
        )

    for factor in factor_cols:
        if factor not in residualized_frame:
            residualized_frame[factor] = np.nan
        residualized_frame[factor] = pd.to_numeric(
            residualized_frame[factor],
            errors="coerce",
        ).astype("float64")
    residualized_frame = residualized_frame[KEYS + factor_cols]

    daily_ic = (
        pd.concat(daily_results, ignore_index=True)
        if daily_results
        else pd.DataFrame()
    )
    diagnostics = {
        "year": start.year,
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "factor_rows": len(factors),
        "barra_rows": len(barra),
        "label_rows": len(labels),
        "factor_barra_rows": len(factor_barra),
        "factor_label_rows": factor_label_rows,
        "residualized_rows": len(residualized_frame),
    }
    return YearResult(
        residualized=residualized_frame,
        daily_ic=daily_ic,
        diagnostics=diagnostics,
        invalid_factors=invalid_factors,
        max_abs_moment=max_abs_moment,
    )


def summarize_ic(daily_ic: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw, matched-raw and neutralized daily IC series."""
    rows: list[dict] = []
    versions = {
        "raw": ("raw_ic", "raw_n"),
        "raw_matched": ("raw_matched_ic", "raw_matched_n"),
        "neutral": ("neutral_ic", "neutral_n"),
    }
    for factor, factor_data in daily_ic.groupby("factor", sort=True):
        for version, (ic_column, n_column) in versions.items():
            values = pd.to_numeric(
                factor_data[ic_column],
                errors="coerce",
            ).dropna()
            count = len(values)
            mean_ic = float(values.mean()) if count else np.nan
            std_ic = float(values.std(ddof=1)) if count > 1 else np.nan
            icir = (
                mean_ic / std_ic
                if count > 1 and np.isfinite(std_ic) and std_ic > 0
                else np.nan
            )
            t_stat = (
                mean_ic / (std_ic / np.sqrt(count))
                if count > 1 and np.isfinite(std_ic) and std_ic > 0
                else np.nan
            )
            sign_win_rate = (
                float((np.sign(values) == np.sign(mean_ic)).mean())
                if count and np.isfinite(mean_ic) and mean_ic != 0
                else np.nan
            )
            rows.append(
                {
                    "factor": factor,
                    "version": version,
                    "mean_ic": mean_ic,
                    "abs_mean_ic": abs(mean_ic)
                    if np.isfinite(mean_ic)
                    else np.nan,
                    "mean_abs_daily_ic": float(values.abs().mean())
                    if count
                    else np.nan,
                    "std_ic": std_ic,
                    "icir": icir,
                    "t_stat": t_stat,
                    "sign_win_rate": sign_win_rate,
                    "positive_rate": float(values.gt(0).mean())
                    if count
                    else np.nan,
                    "effective_days": count,
                    "average_cross_section": float(
                        pd.to_numeric(
                            factor_data.loc[values.index, n_column],
                            errors="coerce",
                        ).mean()
                    )
                    if count
                    else np.nan,
                    "meets_abs_ic_0035": bool(
                        np.isfinite(mean_ic) and abs(mean_ic) >= 0.035
                    ),
                    "meets_abs_ic_0030": bool(
                        np.isfinite(mean_ic) and abs(mean_ic) >= 0.030
                    ),
                }
            )
    return pd.DataFrame(rows)


def _atomic_write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    os.replace(temporary, path)


def _write_csv_with_fallback(frame: pd.DataFrame, path: Path) -> Path:
    """Write CSV atomically; use another name when Excel locks the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    candidates = [
        path,
        path.with_name(f"{path.stem}_latest{path.suffix}"),
        path.with_name(f"{path.stem}_{os.getpid()}{path.suffix}"),
    ]
    for candidate in candidates:
        try:
            os.replace(temporary, candidate)
            return candidate
        except PermissionError:
            continue
    raise PermissionError(
        "IC汇总CSV及备用文件均被占用，请关闭Excel后重试"
    )


def run_factor_test_pipeline(
    factor_path: str | Path = FACTOR_PATH,
    barra_path: str | Path = BARRA_PATH,
    label_path: str | Path = LABEL_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    *,
    factor_columns: Iterable[str] | None = None,
    label_column: str = "LABEL",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    min_cross_section: int = 60,
    min_ic_cross_section: int = 30,
) -> dict:
    """Run the complete merge, neutralization and IC evaluation workflow."""
    factor_path = Path(factor_path).resolve()
    barra_path = Path(barra_path).resolve()
    label_path = Path(label_path).resolve()
    output_dir = Path(output_dir).resolve()
    for path in (factor_path, barra_path, label_path):
        if not path.exists():
            raise FileNotFoundError(path)

    factor_cols = infer_factor_columns(factor_path, factor_columns)
    barra_names = _schema_names(barra_path)
    missing_barra = sorted(set(BARRA_FACTORS).difference(barra_names))
    if missing_barra:
        raise KeyError(f"barra_diy.parquet 缺少中性化列: {missing_barra}")
    label_names = _schema_names(label_path)
    if label_column not in label_names:
        raise KeyError(f"label.parquet 缺少标签列: {label_column}")

    ranges = [
        _date_range(factor_path),
        _date_range(barra_path),
        _date_range(label_path),
    ]
    start = (
        pd.Timestamp(start_date).normalize()
        if start_date is not None
        else max(item[0] for item in ranges)
    )
    end = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else min(item[1] for item in ranges)
    )
    if start > end:
        raise ValueError(f"共同日期范围为空: {start.date()} > {end.date()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    residual_path = output_dir / "factors_neutralized.parquet"
    residual_temporary = residual_path.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    daily_frames: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    invalid_by_year: dict[str, list[str]] = {}
    max_abs_moment = 0.0

    try:
        for year in range(start.year, end.year + 1):
            chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
            chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
            print(f"处理 {chunk_start.date()} 至 {chunk_end.date()}...")
            result = _process_year(
                factor_path,
                barra_path,
                label_path,
                factor_cols,
                BARRA_FACTORS,
                label_column,
                chunk_start,
                chunk_end,
                min_cross_section,
                min_ic_cross_section,
            )
            diagnostics.append(result.diagnostics)
            daily_frames.append(result.daily_ic)
            if result.invalid_factors:
                invalid_by_year[str(year)] = result.invalid_factors
            max_abs_moment = max(
                max_abs_moment,
                result.max_abs_moment,
            )
            if not result.residualized.empty:
                table = pa.Table.from_pandas(
                    result.residualized,
                    preserve_index=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        residual_temporary,
                        table.schema,
                        compression="zstd",
                    )
                writer.write_table(table)
            print(
                f"  factors={result.diagnostics['factor_rows']:,}, "
                f"factor+Barra={result.diagnostics['factor_barra_rows']:,}, "
                f"residualized={result.diagnostics['residualized_rows']:,}"
            )
            del result
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if writer is None or not residual_temporary.exists():
        raise RuntimeError("没有生成有效的残差因子")
    os.replace(residual_temporary, residual_path)

    daily_ic = pd.concat(daily_frames, ignore_index=True)
    daily_ic = daily_ic.sort_values(["factor", "TRADE_DATE"]).reset_index(
        drop=True
    )
    summary = summarize_ic(daily_ic)
    diagnostics_frame = pd.DataFrame(diagnostics)

    daily_path = output_dir / "daily_ic.parquet"
    summary_path = output_dir / "ic_summary.parquet"
    requested_summary_csv_path = output_dir / "ic_summary.csv"
    diagnostics_path = output_dir / "merge_diagnostics.parquet"
    metadata_path = output_dir / "run_metadata.json"
    _atomic_write_frame(daily_ic, daily_path)
    _atomic_write_frame(summary, summary_path)
    _atomic_write_frame(diagnostics_frame, diagnostics_path)
    summary_csv_path = _write_csv_with_fallback(
        summary,
        requested_summary_csv_path,
    )

    metadata = {
        "factor_path": str(factor_path),
        "barra_path": str(barra_path),
        "label_path": str(label_path),
        "label_column": label_column,
        "factor_columns": factor_cols,
        "barra_exposure_columns": BARRA_FACTORS,
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "min_cross_section": min_cross_section,
        "min_ic_cross_section": min_ic_cross_section,
        "max_abs_barra_moment": max_abs_moment,
        "invalid_factors_by_year": invalid_by_year,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nIC 汇总")
    print(summary.to_string(index=False))
    print(f"\n结果已保存到: {output_dir}")
    return {
        "residualized_factors": residual_path,
        "daily_ic": daily_path,
        "ic_summary": summary_path,
        "ic_summary_csv": summary_csv_path,
        "merge_diagnostics": diagnostics_path,
        "metadata": metadata_path,
        "summary": summary,
    }


def build_residualized_factors(
    factor_path: str | Path = FACTOR_PATH,
    barra_path: str | Path = BARRA_PATH,
) -> tuple[pd.DataFrame, list[str], float]:
    """Backward-compatible in-memory residualization for smaller datasets."""
    factor_cols = infer_factor_columns(factor_path)
    raw = restore_keys(pd.read_parquet(factor_path))
    barra = restore_keys(
        pd.read_parquet(
            barra_path,
            columns=KEYS + BARRA_FACTORS,
        )
    )
    _validate_unique_keys(raw, "factors")
    _validate_unique_keys(barra, "Barra")
    base = pd.merge(
        raw[KEYS + factor_cols],
        barra,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    processed: list[pd.Series] = []
    invalid: list[str] = []
    maximum = 0.0
    for factor in factor_cols:
        residualized, moment = _residualize_one_factor(
            base,
            factor,
            BARRA_FACTORS,
            60,
        )
        if residualized.empty:
            invalid.append(factor)
            continue
        processed.append(
            residualized.set_index(KEYS)[factor].rename(factor)
        )
        maximum = max(maximum, moment)
    if not processed:
        raise RuntimeError("没有因子得到有效残差")
    result = pd.concat(processed, axis=1).sort_index().reset_index()
    return result, invalid, maximum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barra 残差化并计算每日 Spearman IC"
    )
    parser.add_argument("--factors", type=Path, default=FACTOR_PATH)
    parser.add_argument("--barra", type=Path, default=BARRA_PATH)
    parser.add_argument("--label", type=Path, default=LABEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--label-column", default="LABEL")
    parser.add_argument("--factor-columns", nargs="+")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--min-cross-section", type=int, default=60)
    parser.add_argument("--min-ic-cross-section", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_factor_test_pipeline(
        args.factors,
        args.barra,
        args.label,
        args.output_dir,
        factor_columns=args.factor_columns,
        label_column=args.label_column,
        start_date=args.start_date,
        end_date=args.end_date,
        min_cross_section=args.min_cross_section,
        min_ic_cross_section=args.min_ic_cross_section,
    )


if __name__ == "__main__":
    main()
