"""Constrained ensemble search across rounds 9--13.

Only 2017--2022 observations determine component directions and optimized
weights.  Sparse period signals have already been tested before filling in
their source rounds; here a missing standardized component is assigned the
same-day neutral value (zero) so the final portfolio retains the full stock
universe.  No future return is used in cross-sectional orthogonalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


KEYS = ["TRADE_DATE", "SECURITY_ID"]
TRAIN_END = pd.Timestamp("2022-12-31")

SOURCES = {
    "round9": (
        "artifacts/round9/filled_strict_test/factors_neutralized.parquet",
        [
            "r9_low_share_issuance",
            "r9_cash_assets",
            "r9_low_debt_growth",
            "r9_cash_accumulation",
            "r9_low_intangible_growth",
        ],
    ),
    "round10": (
        "artifacts/round10/sparse_diagnostic_test/factors_neutralized.parquet",
        [
            "r10_beneish_low_sgi",
            "r10_dechow_low_cash_sales_change",
            "r10_dechow_low_soft_assets",
            "r10_beneish_lvgi_quality_component",
        ],
    ),
    "round11": (
        "artifacts/round11/filled_strict_test/factors_neutralized.parquet",
        [
            "r11_core_profit_surprise_assets",
            "r11_revenue_sue",
            "r11_tax_expense_surprise_assets",
        ],
    ),
    "round12": (
        "artifacts/round12/sparse_diagnostic_test/factors_neutralized.parquet",
        [
            "r12_core_profit_surprise_hl120",
            "r12_revenue_sue_hl120",
            "r12_tax_expense_surprise_hl120",
        ],
    ),
    "round13": (
        "artifacts/round13/sparse_diagnostic_test/factors_neutralized.parquet",
        ["r13_employee_cash_productivity_gap"],
    ),
}

GROUPS = {
    "event": SOURCES["round11"][1],
    "decay": SOURCES["round12"][1],
    "financing": SOURCES["round9"][1],
    "misstatement": SOURCES["round10"][1],
    "employee": SOURCES["round13"][1],
}

FINAL_COLUMNS = [
    "r14_event_equal",
    "r14_event_decay_equal50",
    "r14_diversified_equal4",
    "r14_train_ic_weighted",
    "r14_train_icir_weighted",
    "r14_ridge025_shrink50",
    "r14_ridge100_shrink50",
    "r14_ridge400_shrink50",
    "r14_event_finorth_w10",
    "r14_event_finorth_w20",
    "r14_event_misorth_w10",
    "r14_event_misorth_w20",
    "r14_event_multiorth",
    "r14_orth_ridge100_shrink50",
]


def _read_year(path: Path, columns: list[str], year: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=KEYS + columns,
        filters=[
            ("TRADE_DATE", ">=", pd.Timestamp(year=year, month=1, day=1)),
            ("TRADE_DATE", "<=", pd.Timestamp(year=year, month=12, day=31)),
        ],
    )
    frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
    frame["SECURITY_ID"] = pd.to_numeric(frame["SECURITY_ID"]).astype("int64")
    return frame.drop_duplicates(KEYS, keep="last")


def _robust_z(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    dates = frame["TRADE_DATE"]
    for column in columns:
        value = pd.to_numeric(frame[column], errors="coerce")
        median = value.groupby(dates, sort=False).transform("median")
        deviation = (value - median).abs()
        mad = deviation.groupby(dates, sort=False).transform("median") * 1.4826
        fallback = value.groupby(dates, sort=False).transform("std")
        scale = mad.where(mad.gt(1e-12), fallback.where(fallback.gt(1e-12)))
        z = ((value - median) / scale).clip(-5.0, 5.0)
        out[column] = z.fillna(0.0).astype("float32")
    return out


def _daily_spearman(
    frame: pd.DataFrame, columns: list[str], label: str = "LABEL"
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date, group in frame.groupby("TRADE_DATE", sort=True):
        label_value = pd.to_numeric(group[label], errors="coerce")
        row: dict[str, object] = {"TRADE_DATE": trade_date}
        for column in columns:
            pair = pd.DataFrame(
                {"factor": pd.to_numeric(group[column], errors="coerce"), "label": label_value}
            ).dropna()
            if len(pair) >= 30 and pair["factor"].nunique() > 1:
                ranked = pair.rank(method="average")
                row[column] = ranked["factor"].corr(ranked["label"])
            else:
                row[column] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _orthogonalize(frame: pd.DataFrame, target: str, anchor: str) -> pd.Series:
    x = pd.to_numeric(frame[target], errors="coerce").fillna(0.0)
    a = pd.to_numeric(frame[anchor], errors="coerce").fillna(0.0)
    dates = frame["TRADE_DATE"]
    cov = (x * a).groupby(dates, sort=False).transform("mean") - (
        x.groupby(dates, sort=False).transform("mean")
        * a.groupby(dates, sort=False).transform("mean")
    )
    var = (a * a).groupby(dates, sort=False).transform("mean") - (
        a.groupby(dates, sort=False).transform("mean") ** 2
    )
    beta = cov.div(var.where(var.gt(1e-12))).fillna(0.0)
    return (x - beta * a).astype("float32")


def _normalize_nonnegative(weights: np.ndarray, cap: float = 0.70) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype="float64"), 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    for _ in range(10):
        over = weights > cap
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        under = ~over
        if under.any() and weights[under].sum() > 0:
            weights[under] += excess * weights[under] / weights[under].sum()
    return weights / weights.sum()


def _ridge_weights(daily_ic: pd.DataFrame, columns: list[str], ridge: float) -> np.ndarray:
    values = daily_ic[columns].dropna(how="all").fillna(0.0)
    mean = values.mean().to_numpy(dtype="float64")
    covariance = values.cov().to_numpy(dtype="float64")
    average_variance = float(np.trace(covariance) / max(len(columns), 1))
    regularized = covariance + ridge * max(average_variance, 1e-8) * np.eye(len(columns))
    raw = np.linalg.pinv(regularized) @ mean
    return _normalize_nonnegative(raw)


def _shrink_equal(weights: np.ndarray, amount: float = 0.50) -> np.ndarray:
    equal = np.repeat(1.0 / len(weights), len(weights))
    return _normalize_nonnegative((1.0 - amount) * weights + amount * equal)


def _weighted(frame: pd.DataFrame, columns: list[str], weights: np.ndarray) -> pd.Series:
    values = frame[columns].to_numpy(dtype="float64", copy=False)
    return pd.Series(values @ weights, index=frame.index, dtype="float32")


def _load_components(repo: Path, label_path: Path, year: int) -> pd.DataFrame:
    base_path, base_columns = SOURCES["round11"]
    result = _read_year(repo / base_path, base_columns, year)[KEYS]
    for relative, columns in SOURCES.values():
        source = _read_year(repo / relative, columns, year)
        result = result.merge(source, on=KEYS, how="left", validate="one_to_one")
    label = _read_year(label_path, ["LABEL"], year)
    result = result.merge(label, on=KEYS, how="left", validate="one_to_one")
    return result


def build_block_panel(
    repo: Path, label_path: Path, output_dir: Path, years: list[int]
) -> tuple[dict[str, float], pd.DataFrame]:
    all_components = [column for columns in GROUPS.values() for column in columns]
    component_ic_parts: list[pd.DataFrame] = []
    standardized_by_year: dict[int, pd.DataFrame] = {}
    for year in years:
        raw = _load_components(repo, label_path, year)
        z = _robust_z(raw, all_components)
        standardized = pd.concat([raw[KEYS + ["LABEL"]], z], axis=1)
        if year <= TRAIN_END.year:
            component_ic_parts.append(_daily_spearman(standardized, all_components))
        standardized_by_year[year] = standardized
        print(f"component pass {year}: {len(standardized):,} rows")

    component_ic = pd.concat(component_ic_parts, ignore_index=True)
    directions = {
        column: float(1.0 if component_ic[column].mean() >= 0 else -1.0)
        for column in all_components
    }
    direction_table = pd.DataFrame(
        {
            "factor": all_components,
            "train_mean_ic_raw": [component_ic[c].mean() for c in all_components],
            "direction": [directions[c] for c in all_components],
        }
    )
    direction_table.to_csv(
        output_dir / "component_training_directions.csv", index=False, encoding="utf-8-sig"
    )

    block_dir = output_dir / "block_panel"
    block_dir.mkdir(parents=True, exist_ok=True)
    block_ic_parts: list[pd.DataFrame] = []
    for year in years:
        standardized = standardized_by_year.pop(year)
        block = standardized[KEYS + ["LABEL"]].copy()
        for group, columns in GROUPS.items():
            oriented = standardized[columns].mul(
                pd.Series({column: directions[column] for column in columns})
            )
            block[group] = oriented.mean(axis=1).astype("float32")
        for group in ["financing", "misstatement", "employee"]:
            block[f"{group}_orth"] = _orthogonalize(block, group, "event")
        block.to_parquet(
            block_dir / f"year={year}.parquet", index=False, compression="zstd"
        )
        if year <= TRAIN_END.year:
            block_ic_parts.append(
                _daily_spearman(
                    block,
                    [
                        "event",
                        "decay",
                        "financing",
                        "misstatement",
                        "employee",
                        "financing_orth",
                        "misstatement_orth",
                        "employee_orth",
                    ],
                )
            )
        print(f"block pass {year}: saved")
    block_ic = pd.concat(block_ic_parts, ignore_index=True)
    block_ic.to_parquet(output_dir / "training_block_daily_ic.parquet", index=False)
    return directions, block_ic


def fit_weights(block_ic: pd.DataFrame, output_dir: Path) -> dict[str, dict[str, object]]:
    groups = ["event", "financing", "misstatement", "employee"]
    orth_groups = ["event", "financing_orth", "misstatement_orth", "employee_orth"]
    mean = block_ic[groups].mean().clip(lower=0.0)
    ic_weights = _normalize_nonnegative(mean.to_numpy())
    std = block_ic[groups].std().replace(0.0, np.nan)
    icir_weights = _normalize_nonnegative((mean / std).fillna(0.0).to_numpy())
    fitted: dict[str, dict[str, object]] = {
        "train_ic_weighted": {"columns": groups, "weights": ic_weights.tolist()},
        "train_icir_weighted": {"columns": groups, "weights": icir_weights.tolist()},
    }
    for ridge in [0.25, 1.0, 4.0]:
        weights = _shrink_equal(_ridge_weights(block_ic, groups, ridge), 0.50)
        fitted[f"ridge{ridge:g}_shrink50"] = {
            "columns": groups,
            "weights": weights.tolist(),
        }
    orth_weights = _shrink_equal(_ridge_weights(block_ic, orth_groups, 1.0), 0.50)
    fitted["orth_ridge1_shrink50"] = {
        "columns": orth_groups,
        "weights": orth_weights.tolist(),
    }
    (output_dir / "fitted_weights.json").write_text(
        json.dumps(fitted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return fitted


def build_candidates(output_dir: Path, years: list[int], fitted: dict[str, dict[str, object]]) -> Path:
    output_path = output_dir / "round9_13_optimized_candidates.parquet"
    writer: pq.ParquetWriter | None = None
    for year in years:
        block = pd.read_parquet(output_dir / "block_panel" / f"year={year}.parquet")
        result = block[KEYS].copy()
        result["r14_event_equal"] = block["event"]
        result["r14_event_decay_equal50"] = 0.5 * block["event"] + 0.5 * block["decay"]
        result["r14_diversified_equal4"] = block[
            ["event", "financing", "misstatement", "employee"]
        ].mean(axis=1)
        result["r14_train_ic_weighted"] = _weighted(
            block, **fitted["train_ic_weighted"]
        )
        result["r14_train_icir_weighted"] = _weighted(
            block, **fitted["train_icir_weighted"]
        )
        for ridge, output_name in [
            ("ridge0.25_shrink50", "r14_ridge025_shrink50"),
            ("ridge1_shrink50", "r14_ridge100_shrink50"),
            ("ridge4_shrink50", "r14_ridge400_shrink50"),
        ]:
            result[output_name] = _weighted(block, **fitted[ridge])
        result["r14_event_finorth_w10"] = 0.90 * block["event"] + 0.10 * block["financing_orth"]
        result["r14_event_finorth_w20"] = 0.80 * block["event"] + 0.20 * block["financing_orth"]
        result["r14_event_misorth_w10"] = 0.90 * block["event"] + 0.10 * block["misstatement_orth"]
        result["r14_event_misorth_w20"] = 0.80 * block["event"] + 0.20 * block["misstatement_orth"]
        result["r14_event_multiorth"] = (
            0.75 * block["event"]
            + 0.10 * block["financing_orth"]
            + 0.10 * block["misstatement_orth"]
            + 0.05 * block["employee_orth"]
        )
        result["r14_orth_ridge100_shrink50"] = _weighted(
            block, **fitted["orth_ridge1_shrink50"]
        )
        result[FINAL_COLUMNS] = result[FINAL_COLUMNS].astype("float32")
        table = pa.Table.from_pandas(result[KEYS + FINAL_COLUMNS], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"candidate pass {year}: {len(result):,} rows")
    if writer is not None:
        writer.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/round14"))
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(args.start_year, args.end_year + 1))
    directions, block_ic = build_block_panel(repo, args.label.resolve(), output_dir, years)
    fitted = fit_weights(block_ic, output_dir)
    output = build_candidates(output_dir, years, fitted)
    metadata = {
        "training_period": "2017-01-01/2022-12-31",
        "validation_period": "2023-01-01/2024-12-31",
        "holdout_check_period": "2025-01-01/2026-12-31",
        "directions": directions,
        "candidate_columns": FINAL_COLUMNS,
        "output": str(output),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
