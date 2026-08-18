"""Controlled composites of the strongest fifth-round financial signals.

Weights are fitted from 2017-2022 daily IC only.  The script tests fixed
low-dimensional blends, residual blends orthogonal to net-income SUE,
non-negative ridge/shrinkage portfolios, and small additions to the current
best factor.  Validation and holdout data never enter weight selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dense_q1_gross_profit_factors import robust_daily_zscore
from event_financial_factor_search import KEYS
from third_round_new_factor_optimization import _daily_orthogonal_residual


BASE_DIR = Path(__file__).resolve().parent
COMPONENTS = {
    "net_sue": "dense_q1_net_income_sue",
    "cost": "dense_allq_cost_discipline",
    "surprise_cash": "dense_allq_surprise_growth_cash",
    "rd_efficiency": "dense_q1_rd_growth_efficiency",
    "revenue_growth": "dense_allq_revenue_growth",
}
ANCHOR = "optimized_interaction"


def fit_ridge_weights(daily_ic_path: Path, output: Path) -> dict[str, object]:
    data = pd.read_parquet(daily_ic_path)
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"])
    data = data.loc[
        data["TRADE_DATE"].dt.year.between(2017, 2022)
        & data["factor"].isin(COMPONENTS.values())
    ]
    pivot = data.pivot(index="TRADE_DATE", columns="factor", values="neutral_ic")
    columns = list(COMPONENTS.values())
    pivot = pivot[columns].dropna()
    mean = pivot.mean().to_numpy(dtype=float)
    covariance = pivot.cov().to_numpy(dtype=float)
    scale = float(np.trace(covariance) / len(columns))
    weights: dict[str, dict[str, float]] = {}
    equal = np.full(len(columns), 1.0 / len(columns))
    for ridge in (0.25, 1.0):
        solved = np.linalg.solve(covariance + ridge * scale * np.eye(len(columns)), mean)
        solved = np.clip(solved, 0.0, None)
        solved = solved / solved.sum() if solved.sum() > 0 else equal.copy()
        shrunk = 0.50 * solved + 0.50 * equal
        weights[f"ridge{int(ridge * 100):03d}"] = dict(zip(columns, solved.tolist()))
        weights[f"ridge{int(ridge * 100):03d}_equal50"] = dict(zip(columns, shrunk.tolist()))
    payload = {
        "training_period": "2017-2022",
        "method": "nonnegative ridge on daily neutral IC covariance; 50% variants shrink to equal weight",
        "training_mean_ic": dict(zip(columns, mean.tolist())),
        "weights": weights,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def candidate_names() -> list[str]:
    return [
        "r6_equal3",
        "r6_weighted3_40_30_30",
        "r6_equal4",
        "r6_diversified5",
        "r6_incremental_equal3",
        "r6_incremental_equal4",
        "r6_net_incremental_w10",
        "r6_net_incremental_w20",
        "r6_net_incremental_w30",
        "r6_ridge025",
        "r6_ridge025_equal50",
        "r6_ridge100",
        "r6_ridge100_equal50",
        "r6_anchor_incremental_w05",
        "r6_anchor_incremental_w10",
        "r6_anchor_incremental_w15",
        "r6_anchor_ridge_w05",
        "r6_anchor_ridge_w10",
        "r6_anchor_ridge_w15",
    ]


def _weighted(z: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(z[column] * float(weight) for column, weight in weights.items())


def build_candidates(inputs: pd.DataFrame, parameters: dict[str, object]) -> pd.DataFrame:
    columns = [*COMPONENTS.values(), ANCHOR]
    z = robust_daily_zscore(inputs, columns)
    dates = inputs["TRADE_DATE"]
    net = z[COMPONENTS["net_sue"]]
    cost = z[COMPONENTS["cost"]]
    surprise = z[COMPONENTS["surprise_cash"]]
    rd = z[COMPONENTS["rd_efficiency"]]
    revenue = z[COMPONENTS["revenue_growth"]]
    anchor = z[ANCHOR]

    cost_resid = _daily_orthogonal_residual(cost, net, dates)
    surprise_resid = _daily_orthogonal_residual(surprise, net, dates)
    rd_resid = _daily_orthogonal_residual(rd, net, dates)
    revenue_resid = _daily_orthogonal_residual(revenue, net, dates)
    incremental3 = (cost_resid + surprise_resid + rd_resid) / 3
    incremental4 = (cost_resid + surprise_resid + rd_resid + revenue_resid) / 4

    result = inputs[KEYS].copy()
    result["r6_equal3"] = (net + cost + surprise) / 3
    result["r6_weighted3_40_30_30"] = 0.40 * net + 0.30 * cost + 0.30 * surprise
    result["r6_equal4"] = (net + cost + surprise + rd) / 4
    result["r6_diversified5"] = 0.30 * net + 0.20 * cost + 0.20 * surprise + 0.20 * rd + 0.10 * revenue
    result["r6_incremental_equal3"] = incremental3
    result["r6_incremental_equal4"] = incremental4
    for weight in (0.10, 0.20, 0.30):
        result[f"r6_net_incremental_w{int(weight * 100):02d}"] = (1 - weight) * net + weight * incremental3

    component_z = z[list(COMPONENTS.values())]
    for name, weights in parameters["weights"].items():
        result[f"r6_{name}"] = _weighted(component_z, weights)
    ridge = result["r6_ridge100_equal50"]
    for weight in (0.05, 0.10, 0.15):
        code = int(weight * 100)
        result[f"r6_anchor_incremental_w{code:02d}"] = (1 - weight) * anchor + weight * incremental3
        result[f"r6_anchor_ridge_w{code:02d}"] = (1 - weight) * anchor + weight * ridge
    for factor in candidate_names():
        result[factor] = pd.to_numeric(result[factor], errors="coerce").fillna(0.0).astype("float32")
    return result[KEYS + candidate_names()]


def generate(component_path: Path, anchor_path: Path, parameter_path: Path, output: Path, manifest: Path) -> None:
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    components = pd.read_parquet(component_path, columns=KEYS + list(COMPONENTS.values()))
    anchor = pd.read_parquet(anchor_path, columns=KEYS + [ANCHOR])
    inputs = components.merge(anchor, on=KEYS, how="inner", validate="one_to_one")
    result = build_candidates(inputs, parameters)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    families = [
        "fixed", "fixed", "fixed", "fixed",
        "incremental", "incremental", "net_incremental", "net_incremental", "net_incremental",
        "ridge", "ridge", "ridge", "ridge",
        "anchor_incremental", "anchor_incremental", "anchor_incremental",
        "anchor_ridge", "anchor_ridge", "anchor_ridge",
    ]
    pd.DataFrame({"factor": candidate_names(), "family": families}).to_csv(
        manifest, index=False, encoding="utf-8-sig"
    )


def summarize(daily_ic_path: Path, manifest_path: Path, output: Path) -> None:
    data = pd.read_parquet(daily_ic_path)
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"])
    rows = []
    for factor, group in data.groupby("factor"):
        train = group.loc[group["TRADE_DATE"].dt.year.between(2017, 2022)]
        yearly = train.groupby(train["TRADE_DATE"].dt.year)["neutral_ic"].mean()
        rows.append({
            "factor": factor,
            "full_ic": group["neutral_ic"].mean(),
            "train_2017_2022_ic": train["neutral_ic"].mean(),
            "validation_2023_2024_ic": group.loc[group["TRADE_DATE"].dt.year.between(2023, 2024), "neutral_ic"].mean(),
            "holdout_2025_2026_ic": group.loc[group["TRADE_DATE"].dt.year.between(2025, 2026), "neutral_ic"].mean(),
            "positive_train_years": int(yearly.gt(0).sum()),
        })
    result = pd.DataFrame(rows).merge(pd.read_csv(manifest_path), on="factor", how="left")
    result["selected_in_family"] = False
    for _, positions in result.groupby("family").groups.items():
        subset = result.loc[positions]
        stable = subset.loc[subset["positive_train_years"].ge(5)]
        pool = stable if not stable.empty else subset
        result.loc[pool["train_2017_2022_ic"].idxmax(), "selected_in_family"] = True
    result.sort_values("full_ic", ascending=False).to_csv(output, index=False, encoding="utf-8-sig")


def main() -> None:
    root = BASE_DIR / "新测试结果" / "第六轮受控复合"
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--component-daily-ic", type=Path, default=BASE_DIR / "新测试结果" / "第五轮旧高IC族稠密优化" / "strict_test" / "daily_ic.parquet")
    parser.add_argument("--components", type=Path, default=BASE_DIR / "新测试结果" / "第五轮旧高IC族稠密优化" / "round5_selected_with_existing_six.parquet")
    parser.add_argument("--anchor", type=Path, default=BASE_DIR / "新测试结果" / "第二轮优化" / "round2_selected.parquet")
    parser.add_argument("--parameters", type=Path, default=root / "round6_training_weights.json")
    parser.add_argument("--output", type=Path, default=root / "round6_grid.parquet")
    parser.add_argument("--manifest", type=Path, default=root / "round6_manifest.csv")
    parser.add_argument("--daily-ic", type=Path, default=root / "strict_test" / "daily_ic.parquet")
    parser.add_argument("--report", type=Path, default=root / "round6_selection_report.csv")
    args = parser.parse_args()
    if args.fit:
        fit_ridge_weights(args.component_daily_ic.resolve(), args.parameters.resolve())
    elif args.summarize:
        summarize(args.daily_ic.resolve(), args.manifest.resolve(), args.report.resolve())
    else:
        generate(args.components.resolve(), args.anchor.resolve(), args.parameters.resolve(), args.output.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
