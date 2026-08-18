"""Round 22: out-of-sample ridge factors using non-profit financial signals.

2017-2020 selects ridge strength on 2021-2022 from three fixed choices.
The selected specification is refit on 2017-2022 and frozen for 2023-2026.
No profitability level, earnings growth, earnings SUE, or prior composite is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from round9_13_ensemble_optimization import KEYS, _robust_z


RAW = [
    "quarterly_f_score", "mohanram_g_score", "cfo_sue", "accrual_quality",
    "asset_growth", "investment_to_assets", "receivable_abnormal_growth",
    "inventory_abnormal_growth", "management_mispricing_score",
    "accrual_nonrecurring_score",
]
BLOCK = ["financing", "misstatement", "employee"]
CONTRACT = ["r17_contract_funding_change_assets"]
REPORT = [
    "r20_reporting_timeliness_level", "r20_low_revision_count",
    "r20_reporting_timeliness_yoy",
]
BASE = RAW + BLOCK + CONTRACT + REPORT
INTERACTIONS = {
    "joint_financing_fscore": ("financing", "quarterly_f_score"),
    "joint_asset_investment": ("asset_growth", "investment_to_assets"),
    "joint_accrual_misstatement": ("accrual_quality", "misstatement"),
    "joint_financing_contract": ("financing", CONTRACT[0]),
    "joint_timing_revision": (REPORT[0], REPORT[1]),
    "joint_cfo_fscore": ("cfo_sue", "quarterly_f_score"),
}
ALPHAS = [0.1, 1.0, 10.0]


def _read_year(path: Path, columns: list[str], year: int) -> pd.DataFrame:
    filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
    frame = pd.read_parquet(path, columns=columns, filters=filters)
    frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
    return frame


def _features(args: argparse.Namespace, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _read_year(args.panel, KEYS + RAW, year)
    blocks = pd.read_parquet(args.blocks / f"year={year}.parquet", columns=KEYS + BLOCK)
    contract = _read_year(args.contract, KEYS + CONTRACT, year)
    report = _read_year(args.reporting, KEYS + REPORT, year)
    label = _read_year(args.label, KEYS + ["LABEL"], year)
    for frame in (blocks,):
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
    data = raw.merge(blocks, on=KEYS, how="left", validate="one_to_one")
    data = data.merge(contract, on=KEYS, how="left", validate="one_to_one")
    data = data.merge(report, on=KEYS, how="left", validate="one_to_one")
    data = data.merge(label, on=KEYS, how="left", validate="one_to_one")
    data[BASE] = data[BASE].replace([np.inf, -np.inf], np.nan)
    data[BASE] = data[BASE].fillna(data.groupby("TRADE_DATE", sort=False)[BASE].transform("median")).fillna(0.0)
    z = _robust_z(data, BASE).clip(-5.0, 5.0)
    extended = z.copy()
    for name, (left, right) in INTERACTIONS.items():
        extended[name] = z[left] * z[right]
    interaction_names = list(INTERACTIONS)
    interaction_z = _robust_z(pd.concat([data[KEYS], extended[interaction_names]], axis=1), interaction_names).clip(-5.0, 5.0)
    extended[interaction_names] = interaction_z
    target = data.groupby("TRADE_DATE", sort=False)["LABEL"].rank(pct=True) - 0.5
    return data[KEYS], pd.concat([extended, target.rename("TARGET")], axis=1)


def _moments(features: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    valid = features["TARGET"].notna()
    x = features.loc[valid, columns].to_numpy(dtype="float64", copy=False)
    y = features.loc[valid, "TARGET"].to_numpy(dtype="float64", copy=False)
    return x.T @ x, x.T @ y, len(y)


def _fit(xtx: np.ndarray, xty: np.ndarray, count: int, alpha: float) -> np.ndarray:
    covariance = xtx / max(count, 1)
    response = xty / max(count, 1)
    return np.linalg.pinv(covariance + alpha * np.eye(len(response))) @ response


def _mean_daily_ic(keys: pd.DataFrame, score: np.ndarray, target: pd.Series) -> float:
    frame = keys.copy()
    frame["score"] = score
    frame["target"] = target.to_numpy()
    frame["score_rank"] = frame.groupby("TRADE_DATE", sort=False)["score"].rank(pct=True)
    frame["target_rank"] = frame.groupby("TRADE_DATE", sort=False)["target"].rank(pct=True)
    values = frame.groupby("TRADE_DATE", sort=False).apply(
        lambda g: g["score_rank"].corr(g["target_rank"]), include_groups=False
    )
    return float(values.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reporting", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    linear_columns = BASE
    extended_columns = BASE + list(INTERACTIONS)
    specs = {"linear": linear_columns, "interactions": extended_columns}
    train_moments = {name: [np.zeros((len(cols), len(cols))), np.zeros(len(cols)), 0] for name, cols in specs.items()}
    full_moments = {name: [np.zeros((len(cols), len(cols))), np.zeros(len(cols)), 0] for name, cols in specs.items()}
    validation: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for year in range(2017, 2023):
        keys, features = _features(args, year)
        for name, columns in specs.items():
            xtx, xty, count = _moments(features, columns)
            full_moments[name][0] += xtx
            full_moments[name][1] += xty
            full_moments[name][2] += count
            if year <= 2020:
                train_moments[name][0] += xtx
                train_moments[name][1] += xty
                train_moments[name][2] += count
        if year >= 2021:
            validation.append((keys, features))
        print(f"training pass {year}: {len(keys):,} rows")

    selected: dict[str, float] = {}
    validation_scores: list[dict[str, object]] = []
    for name, columns in specs.items():
        xtx, xty, count = train_moments[name]
        best_alpha, best_ic = ALPHAS[0], -np.inf
        for alpha in ALPHAS:
            coef = _fit(xtx, xty, count, alpha)
            daily_parts = []
            for keys, features in validation:
                score = features[columns].to_numpy(dtype="float64", copy=False) @ coef
                daily_parts.append(_mean_daily_ic(keys, score, features["TARGET"]))
            ic = float(np.mean(daily_parts))
            validation_scores.append({"specification": name, "alpha": alpha, "validation_mean_ic": ic})
            if ic > best_ic:
                best_alpha, best_ic = alpha, ic
        selected[name] = best_alpha

    coefficients = {}
    for name, columns in specs.items():
        xtx, xty, count = full_moments[name]
        coefficients[name] = _fit(xtx, xty, count, selected[name])

    output_parts = []
    for year in range(2017, 2027):
        keys, features = _features(args, year)
        result = keys.copy()
        result["r22_nonprofit_ridge_linear"] = features[linear_columns].to_numpy(dtype="float64", copy=False) @ coefficients["linear"]
        result["r22_nonprofit_ridge_interactions"] = features[extended_columns].to_numpy(dtype="float64", copy=False) @ coefficients["interactions"]
        output_parts.append(result)
        print(f"prediction pass {year}: {len(result):,} rows")
    output = pd.concat(output_parts, ignore_index=True).sort_values(KEYS)
    factor_columns = ["r22_nonprofit_ridge_linear", "r22_nonprofit_ridge_interactions"]
    output[factor_columns] = output[factor_columns].astype("float32")
    output.to_parquet(args.output_dir / "round22_nonprofit_ridge_candidates.parquet", index=False, compression="zstd")
    pd.DataFrame(validation_scores).to_csv(args.output_dir / "ridge_validation.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "train_for_alpha": "2017-2020", "validation_for_alpha": "2021-2022",
        "refit": "2017-2022", "frozen_evaluation": "2023-2026",
        "alphas": ALPHAS, "selected_alpha": selected,
        "features": specs, "uses_profitability_or_earnings_sue": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
