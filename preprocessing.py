# preprocessing.py
# ----------------------------------------------------------
# Handles input data cleaning for load profile & assumptions
# ----------------------------------------------------------

import pandas as pd
from helpers import normalize_columns, normalize_text, convert_to_float, convert_percentages, log

def preprocess_inputs(df_load_profile: pd.DataFrame, df_assumptions: pd.DataFrame, df_assumptions_capex: pd.DataFrame):
    """
    Clean and standardize the input DataFrames before modeling.

    Steps:
    1. Normalize column names.
    2. Convert numeric text fields (like Power & Wind generation) to float.
    3. Normalize assumptions and convert percentage units to decimals.

    Parameters
    ----------
    df_load_profile : pd.DataFrame
        Raw load profile (solar + wind generation).
    df_assumptions : pd.DataFrame
        Raw assumptions data.

    Returns
    -------
    df_load_profile : pd.DataFrame
        Cleaned load profile.
    df_assumptions : pd.DataFrame
        Cleaned assumptions (only 'assumptions' & 'value' columns, float dtype).
    """

    # ---------------------------
    # 1️ Load profile cleaning
    # ---------------------------
    df_load_profile = normalize_columns(df_load_profile)

    # Convert all power/wind columns to float safely
    cols_to_convert = [
        c for c in df_load_profile.columns 
        if "power_mwac" in c or "wind_gen" in c
    ]
    df_load_profile = convert_to_float(df_load_profile, cols_to_convert)

    log(" Solar–Wind Profile cleaned successfully.")
    # log(df_load_profile.dtypes)

    # ---------------------------
    # 2️ Assumptions cleaning
    # ---------------------------
    df_assumptions = normalize_columns(df_assumptions)
    if "assumptions" in df_assumptions.columns:
        df_assumptions["assumptions"] = df_assumptions["assumptions"].apply(normalize_text)

    # Convert % → decimal
    df_assumptions = convert_percentages(
        df_assumptions, value_col="value", unit_col="unit"
    )

    # Keep only assumption/value cols & ensure float dtype
    df_assumptions = df_assumptions[["assumptions", "value"]].copy()
    df_assumptions["value"] = pd.to_numeric(df_assumptions["value"], errors="coerce")

    log(" Assumptions cleaned successfully.")
    
    # ---------------------------
    # 3 Capex assumptions cleaning
    # ---------------------------
    df_assumptions_capex = normalize_columns(df_assumptions_capex)

    # Ensure numeric value column
    df_assumptions_capex["value"] = pd.to_numeric(
        df_assumptions_capex["value"], errors="coerce"
    )

    # Group by 'type' and sum 'value'
    df_assumptions_capex = (
        df_assumptions_capex
        .groupby("type", as_index=False)["value"]
        .sum()
    )

    # Convert Rs → Cr (divide by 10^7)
    df_assumptions_capex["value"] = df_assumptions_capex["value"] / 1e7

    # Rename columns to match your unified assumption schema
    df_assumptions_capex.rename(
        columns={"type": "assumptions"}, inplace=True
    )

    log(" CAPEX assumptions cleaned successfully.")

    return df_load_profile, df_assumptions, df_assumptions_capex
