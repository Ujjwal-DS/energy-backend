# utils.py
# Utility functions for dummy output creation and tender mocks

import os
import pandas as pd
import numpy as np
from helpers import log

# Map of which numeric columns can be varied per file
COLUMN_VARIATION_RULES = {
    "capacity_mix": ["capacity"],
    "project_free_cashflows": ["capex", "fcfe"],
    "re_energy_generation_profile": ["generation"],
    "sensitivity_analysis": ["IRR(%)", "NPV(Cr)"],
    "bess_schedule": [
        "opening_balance_bess",
        "closing_balance_bess",
        "charge_mw",
        "discharge_mw",
        "soc_percent",
        "spc_percent",
    ],
    "metrics": ["value"],  # metrics.csv: columns -> metric, value
}

def _vary_series(x: pd.Series, variation: float) -> pd.Series:
    noise = np.random.uniform(1 - variation, 1 + variation, size=len(x))
    return x.astype(float) * noise

def generate_dummy_outputs(reference_dir: str, output_dir: str, variation: float):
    """
    Create dummy copies of CSVs from FDRE reference, applying small variations.
    Returns:
      outputs_dict: {basename: DataFrame}
      metrics_dict: {metric_name: value}
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs_dict = {}
    metrics_dict = {}

    if not os.path.isdir(reference_dir):
        log(f"[Dummy] Reference dir missing: {reference_dir}")
        return outputs_dict, metrics_dict

    for filename in os.listdir(reference_dir):
        if not filename.endswith(".csv"):
            continue

        ref_path = os.path.join(reference_dir, filename)
        base_name = os.path.splitext(filename)[0].lower()
        out_path = os.path.join(output_dir, filename)
        try:
            df = pd.read_csv(ref_path)

            if base_name == "metrics":
                # vary only "value"
                if "value" in df.columns:
                    if df["value"].dtype.kind in "iufc":
                        df["value"] = _vary_series(df["value"], variation)
                df.to_csv(out_path, index=False)
                outputs_dict[base_name] = df.copy()

                # also build metrics_dict from this CSV
                for _, r in df.iterrows():
                    m = str(r.get("metric", "")).strip()
                    v = float(r.get("value", 0.0))
                    if m:
                        metrics_dict[m] = v
                log(f"[Dummy] metrics.csv varied & saved.")
                continue

            # other CSVs
            allowed_cols = COLUMN_VARIATION_RULES.get(base_name, [])
            if allowed_cols:
                for col in allowed_cols:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = _vary_series(df[col], variation)

            df.to_csv(out_path, index=False)
            outputs_dict[base_name] = df.copy()
            log(f"[Dummy] {filename} saved with variations in {allowed_cols}")
        except Exception as e:
            log(f"[Dummy][Warn] Failed {filename}: {e}")

    log(f"Dummy outputs generated in {output_dir}")
    return outputs_dict, metrics_dict
