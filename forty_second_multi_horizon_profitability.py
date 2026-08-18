"""Round 42: fixed multi-horizon profitability confirmation factors.

Quarterly and trailing-four-quarter measurements of the same accounting
concept are standardized cross-sectionally and averaged.  This is a dense
state construction: it does not use event decay, ranks, labels, or any prior
best/composite factor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from round9_13_ensemble_optimization import KEYS, _robust_z


INPUTS = [
    "dense_lit_q_gp_assets", "dense_lit_ttm_gp_assets",
    "dense_lit_q_op_assets", "dense_lit_ttm_op_assets",
    "dense_lit_qcfoa", "dense_lit_ttm_cfoa",
]
OUTPUTS = [
    "r42_gp_multi_horizon_equal",
    "r42_op_multi_horizon_equal",
    "r42_cfo_multi_horizon_equal",
    "r42_gp_op_multi_horizon_equal",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_parquet(args.panel, columns=KEYS).drop_duplicates(KEYS)
    dense = pd.read_parquet(args.dense, columns=KEYS + INPUTS).drop_duplicates(KEYS)
    for frame in (panel, dense):
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
        frame["SECURITY_ID"] = pd.to_numeric(frame["SECURITY_ID"]).astype("int64")
    data = panel.merge(dense, on=KEYS, how="left", validate="one_to_one")
    data = data.sort_values(["SECURITY_ID", "TRADE_DATE"])
    # The dense source may end before the label calendar.  Financial values
    # are PIT states, so preserve the last already-disclosed observation.
    data[INPUTS] = data.groupby("SECURITY_ID", sort=False)[INPUTS].ffill()
    medians = data.groupby("TRADE_DATE", sort=False)[INPUTS].transform("median")
    data[INPUTS] = data[INPUTS].fillna(medians).fillna(0.0)
    z = _robust_z(data, INPUTS)
    out = data[KEYS].copy()
    out["r42_gp_multi_horizon_equal"] = 0.5 * z["dense_lit_q_gp_assets"] + 0.5 * z["dense_lit_ttm_gp_assets"]
    out["r42_op_multi_horizon_equal"] = 0.5 * z["dense_lit_q_op_assets"] + 0.5 * z["dense_lit_ttm_op_assets"]
    out["r42_cfo_multi_horizon_equal"] = 0.5 * z["dense_lit_qcfoa"] + 0.5 * z["dense_lit_ttm_cfoa"]
    out["r42_gp_op_multi_horizon_equal"] = (
        z["dense_lit_q_gp_assets"] + z["dense_lit_ttm_gp_assets"]
        + z["dense_lit_q_op_assets"] + z["dense_lit_ttm_op_assets"]
    ) / 4.0
    out[OUTPUTS] = out[OUTPUTS].astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(KEYS).to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(out):,} rows and {len(OUTPUTS)} factors")


if __name__ == "__main__":
    main()
