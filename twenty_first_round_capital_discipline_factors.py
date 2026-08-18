"""Round 21: capital allocation, financing, and accounting discipline factors."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from round9_13_ensemble_optimization import KEYS, _robust_z


RAW_COLUMNS = [
    "quarterly_f_score",
    "accrual_quality",
    "asset_growth",
    "investment_to_assets",
    "management_mispricing_score",
]
BLOCK_COLUMNS = ["financing", "misstatement", "employee"]
CONTRACT = "r17_contract_funding_change_assets"


def _weighted(z: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("weights must sum to one")
    return sum(weight * z[column] for column, weight in weights.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_parquet(args.panel, columns=KEYS + RAW_COLUMNS)
    blocks = pd.read_parquet(args.blocks, columns=KEYS + BLOCK_COLUMNS)
    contract = pd.read_parquet(args.contract, columns=KEYS + [CONTRACT])
    for frame in (raw, blocks, contract):
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
    data = raw.merge(blocks, on=KEYS, how="left", validate="one_to_one").merge(
        contract, on=KEYS, how="left", validate="one_to_one"
    )
    inputs = RAW_COLUMNS + BLOCK_COLUMNS + [CONTRACT]
    data[inputs] = data[inputs].fillna(
        data.groupby("TRADE_DATE", sort=False)[inputs].transform("median")
    ).fillna(0.0)
    # Directions are economic and fixed before testing.
    data["low_asset_growth"] = -data["asset_growth"]
    data["low_investment"] = -data["investment_to_assets"]
    data["low_mispricing"] = -data["management_mispricing_score"]
    oriented = [
        "quarterly_f_score", "accrual_quality", "low_asset_growth", "low_investment",
        "low_mispricing", "financing", "misstatement", "employee", CONTRACT,
    ]
    z = _robust_z(data, oriented)
    interaction = pd.DataFrame(index=data.index)
    interaction["financing_fscore_joint"] = z["financing"] * z["quarterly_f_score"]
    interaction["financing_contract_joint"] = z["financing"] * z[CONTRACT]
    interaction_z = _robust_z(
        pd.concat([data[KEYS], interaction], axis=1), list(interaction.columns)
    )
    z = pd.concat([z, interaction_z], axis=1)

    definitions = {
        "r21_capital_discipline_equal": {
            "financing": 0.35, "quarterly_f_score": 0.25,
            "low_asset_growth": 0.20, "low_investment": 0.20,
        },
        "r21_financing_accounting_quality": {
            "financing": 0.35, "misstatement": 0.25,
            "quarterly_f_score": 0.25, "accrual_quality": 0.15,
        },
        "r21_conservative_balance_sheet": {
            "financing": 0.40, "low_asset_growth": 0.25,
            "low_investment": 0.15, CONTRACT: 0.20,
        },
        "r21_structural_quality_breadth": {
            "financing": 0.25, "misstatement": 0.20, "employee": 0.10,
            "quarterly_f_score": 0.20, "accrual_quality": 0.10,
            "low_asset_growth": 0.10, "low_investment": 0.05,
        },
        "r21_financing_fscore_interaction": {
            "financing": 0.45, "quarterly_f_score": 0.35,
            "financing_fscore_joint": 0.20,
        },
        "r21_financing_contract_confirmation": {
            "financing": 0.50, CONTRACT: 0.30,
            "financing_contract_joint": 0.20,
        },
        "r21_accounting_conservatism": {
            "misstatement": 0.35, "accrual_quality": 0.25,
            "low_mispricing": 0.20, "quarterly_f_score": 0.20,
        },
    }
    result = data[KEYS].copy()
    for factor, weights in definitions.items():
        result[factor] = _weighted(z, weights)
    columns = list(definitions)
    result[columns] = result[columns].astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(result):,} rows and {len(columns)} capital-discipline factors")


if __name__ == "__main__":
    main()
