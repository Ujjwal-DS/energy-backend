# helpers.py
# -------------------------------------------------------------------
# Common helper utilities for data cleaning, normalization, and lookup
# -------------------------------------------------------------------

import re
import numpy as np
import pandas as pd
from warnings import filterwarnings
from datetime import datetime, timedelta
from pyxirr import xirr

filterwarnings("ignore")

# -------------------------------------------------------------------
# Text / Column Normalization
# -------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """
    Normalize text to lowercase, remove special characters, 
    replace spaces and multiple underscores with a single underscore.
    """
    s = str(s).strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization to all column names in a DataFrame."""
    df.columns = [normalize_text(c) for c in df.columns]
    return df


# -------------------------------------------------------------------
# Numeric Conversion Helpers
# -------------------------------------------------------------------
def convert_to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert listed columns to float, replacing invalid entries ('-', '', nan, etc.) with 0.
    Nonexistent columns are ignored.
    """
    for col in cols:
        if col not in df.columns:
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace(['-', '', 'nan', 'None', 'NaN', 'null'], '0')
        )

        # use pandas to_numeric with errors='coerce' for safer casting
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df


def convert_percentages(df: pd.DataFrame, value_col: str = 'value', unit_col: str = 'unit') -> pd.DataFrame:
    """
    Convert percentage values to decimals for rows where unit == '%'.
    Example: 45% → 0.45
    """
    if unit_col in df.columns and value_col in df.columns:
        mask = df[unit_col].astype(str).str.strip() == '%'
        df.loc[mask, value_col] = df.loc[mask, value_col].astype(float) / 100
    return df


# -------------------------------------------------------------------
# Assumption Extraction Utility
# -------------------------------------------------------------------
def get_val(
    df: pd.DataFrame,
    name: str,
    key_col: str = "assumptions",
    val_col: str = "value",
    single: bool = True
):
    """
    Retrieve value(s) for a given assumption name from a DataFrame.
    If single=True → return first match as float or NaN.
    If single=False → return list of all matching values.
    """
    if key_col not in df.columns or val_col not in df.columns:
        raise KeyError(f"Expected columns '{key_col}' and '{val_col}' not found in DataFrame")

    # normalize assumption names for case-insensitive match
    mask = df[key_col].astype(str).str.lower().str.strip() == name.lower()
    vals = df.loc[mask, val_col].dropna().tolist()

    if not vals:
        return np.nan if single else []

    # Convert to float safely
    try:
        floats = [float(v) for v in vals]
    except Exception:
        floats = vals  # fallback if non-numeric

    return floats[0] if single else floats

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

