# logic.py
# -----------------------------------------------------------
# Core model functions for GH2 FDRE computation
# -----------------------------------------------------------

import numpy as np
import pandas as pd
from datetime import datetime
from pyxirr import xirr
from helpers import get_val, normalize_columns, log

# ============================================================
#  Function: generate_main_dataframe
# ============================================================
def generate_main_dataframe(df_load_profile: pd.DataFrame, df_assumptions: pd.DataFrame) -> pd.DataFrame:

    """Generate GH2 master table (solar + wind + BESS logic)."""
    

    peak_hrs = get_val(df_assumptions, "peak_hrs", single=False)
    log(f"peak_hrs: {peak_hrs}")

    # Other assumption parameters
    solar_capacity       = get_val(df_assumptions, "solar_capacity")
    wind_capacity        = get_val(df_assumptions, "wind_capacity")
    max_capacity_bess    = get_val(df_assumptions, "max_capacity_bess")
    round_trip_eff_bess  = get_val(df_assumptions, "round_trip_eff_bess")
    aux_consumption      = get_val(df_assumptions, "aux_consumption")
    dtl_losses           = get_val(df_assumptions, "dtl_losses")
    bid_capacity_peak    = get_val(df_assumptions, "bid_capacity_peak")
    bid_capacity_offpeak = get_val(df_assumptions, "bid_capacity_offpeak")

    # ----------------------------------------------------------------
    #  STEP 2: Base structure — month × day × slot
    # ----------------------------------------------------------------
    df = df_load_profile.copy()
    df = df.sort_values(by=["month", "day", "min"]).reset_index(drop=True)

    # Ensure column dtypes are correct
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour"] = pd.to_datetime(df["min"], format="%H:%M:%S").dt.hour + 1
    df["slot_no"] = df.index + 1  # running slot index

    # ----------------------------------------------------------------
    #  STEP 3: Generation calculations (vectorized)
    # ----------------------------------------------------------------
    df["solar"] = solar_capacity * df["power_mwac"] / 4
    df["wind"] = wind_capacity * df["wind_generation"] / 4
    df["total_generation"] = (df["solar"] + df["wind"]) * (1 - aux_consumption) * (1 - dtl_losses)

    # ----------------------------------------------------------------
    #  STEP 4: Peak / Off-peak and Discom requirement
    # ----------------------------------------------------------------
    df["peak_offpeak"] = np.where(df["hour"].isin(peak_hrs), "Peak", "Off-Peak")
    df["discom_requirement"] = np.where(
        df["peak_offpeak"] == "Peak", bid_capacity_peak, bid_capacity_offpeak
    )

    # ----------------------------------------------------------------
    #  STEP 5: Surplus / Shortfall
    # ----------------------------------------------------------------
    df["surplus_shortfall"] = df["total_generation"] - df["discom_requirement"]

    # ----------------------------------------------------------------
    #  STEP 6: Pre-allocate BESS arrays for performance
    # ----------------------------------------------------------------
    n = len(df)
    opening = np.zeros(n)
    energy_avail_chg = np.zeros(n)
    energy_used = np.zeros(n)
    total_energy = np.zeros(n)
    extractable = np.zeros(n)
    discharge = np.zeros(n)
    closing = np.zeros(n)
    energy_left = np.zeros(n)
    discom_met = np.zeros(n)
    shortfall = np.zeros(n)
    availability = np.zeros(n)
    status = np.empty(n, dtype=object)

    # ----------------------------------------------------------------
    #  STEP 7: Iterative BESS logic (chronological dependency)
    # ----------------------------------------------------------------
    for i in range(n):
        prev_closing = closing[i - 1] if i > 0 else 0
        surplus = df.at[i, "surplus_shortfall"]

        # Determine charge/discharge status
        if prev_closing <= 0:
            status[i] = "No Action"
        elif surplus <= 0:
            status[i] = "Discharge"
        else:
            status[i] = "Charge"

        # Energy available for charging
        energy_avail_chg[i] = max(0, surplus)

        # Energy used for charging
        if prev_closing == 0:
            energy_used[i] = min(max_capacity_bess, energy_avail_chg[i])
        elif prev_closing == max_capacity_bess:
            energy_used[i] = 0
        elif prev_closing < max_capacity_bess:
            energy_used[i] = energy_avail_chg[i]
        else:
            energy_used[i] = 0

        # Total energy in battery
        total_energy[i] = min(max_capacity_bess, prev_closing + energy_used[i])

        # Extractable energy (apply efficiency)
        extractable[i] = total_energy[i] * round_trip_eff_bess

        # Actual discharge
        discharge[i] = min(-surplus, extractable[i]) if surplus <= 0 else 0

        # Closing balance
        if discharge[i] > 0:
            closing[i] = total_energy[i] - discharge[i] / round_trip_eff_bess
        else:
            closing[i] = total_energy[i] - discharge[i]

        # Energy left after charging
        energy_left[i] = max(0, (prev_closing + energy_avail_chg[i]) - max_capacity_bess)

        # Total DISCOM demand met
        discom_met[i] = (
            df.at[i, "discom_requirement"]
            if surplus >= 0
            else df.at[i, "total_generation"] + discharge[i]
        )

        # Shortfall
        shortfall[i] = (
            0
            if abs(discom_met[i]) >= df.at[i, "discom_requirement"]
            else discom_met[i] - df.at[i, "discom_requirement"]* 0.9 
        )

        # Availability %
        availability[i] = discom_met[i] / df.at[i, "discom_requirement"]

        # Opening balance for clarity
        opening[i] = prev_closing

    # ----------------------------------------------------------------
    #  STEP 8: Assign arrays back to DataFrame
    # ----------------------------------------------------------------
    df["opening_balance_bess"] = opening
    df["status_charge_discharge"] = status
    df["energy_available_for_charging"] = energy_avail_chg
    df["energy_used_for_charging"] = energy_used
    df["total_energy_in_battery"] = total_energy
    df["total_extractable_energy"] = extractable
    df["actual_discharge"] = discharge
    df["closing_balance_bess"] = closing
    df["energy_left_after_charging"] = energy_left
    df["total_discom_demand_met"] = discom_met
    df["shortfall"] = shortfall
    df["availability_pct"] = availability * 100

    # STEP 9: Add 15-min intervals (HH:MM format)
    df["minute_num"] = df.groupby(["month", "day"]).cumcount() * 15
    df["minute_num"] = df["minute_num"] % 1440
    df["minute"] = (
        (df["minute_num"] // 60).astype(int).astype(str).str.zfill(2) + ":" +
        (df["minute_num"] % 60).astype(int).astype(str).str.zfill(2)
    )
    df = df.drop(columns=["minute_num"])
    
    # ----------------------------------------------------------------
    #  STEP 10: Final columns (includes day)
    # ----------------------------------------------------------------
    final_cols = [
        "month",
        "day", 
        "slot_no",
        "hour",
        "minute",
        "peak_offpeak",
        "solar",
        "wind",
        "total_generation",
        "discom_requirement",
        "surplus_shortfall",
        "status_charge_discharge",
        "opening_balance_bess",
        "energy_available_for_charging",
        "energy_used_for_charging",
        "total_energy_in_battery",
        "total_extractable_energy",
        "actual_discharge",
        "closing_balance_bess",
        "energy_left_after_charging",
        "total_discom_demand_met",
        "shortfall",
        "availability_pct",
    ]

    return df[final_cols].copy()


# ============================================================
#  Function: generate_results_summary (Normalized Columns)
# ============================================================

def generate_results_summary(df_main: pd.DataFrame, df_assumptions: pd.DataFrame, df_assumptions_capex: pd.DataFrame, df_monthly_shortfall: pd.DataFrame) -> pd.DataFrame:
    
    """
    Summarize KPIs such as CUF, availability, and CAPEX.
    """

    bid_capacity_peak    = get_val(df_assumptions, "bid_capacity_peak")
    bid_capacity_offpeak = get_val(df_assumptions, "bid_capacity_offpeak")
    solar_capacity       = get_val(df_assumptions, "solar_capacity")
    wind_capacity        = get_val(df_assumptions, "wind_capacity")
    max_capacity_bess    = get_val(df_assumptions, "max_capacity_bess")
    capex_solar = get_val(df_assumptions_capex, "capex_solar")
    capex_wind  = get_val(df_assumptions_capex, "capex_wind")
    capex_bess  = get_val(df_assumptions_capex, "capex_bess")


    # ----------------------------------------------------------------
    #  Market Sales (Peak / Off-Peak)
    # ----------------------------------------------------------------
    market_sales_peak = (
        df_main.loc[df_main["peak_offpeak"] == "Peak", "energy_left_after_charging"].sum()
    )
    market_sales_offpeak = (
        df_main.loc[df_main["peak_offpeak"] == "Off-Peak", "energy_left_after_charging"].sum()
    )

    # ----------------------------------------------------------------
    #  CUF (Capacity Utilization Factor)
    # ----------------------------------------------------------------
    numerator = df_main["total_discom_demand_met"].sum()
    denominator = (
        (bid_capacity_peak * 24 * 365) + (bid_capacity_offpeak * 24 * 365)
    )
    cuf = (numerator / denominator) * 100 if denominator > 0 else 0

    # ----------------------------------------------------------------
    #  Penalty (total monthly shortfall × 0)
    # ----------------------------------------------------------------
    total_shortfall = (
        df_monthly_shortfall["penalty"].sum() if "penalty" in df_monthly_shortfall.columns else 0
    )
    penalty = total_shortfall * 0  # as per your current rule

    # ----------------------------------------------------------------
    #  Availability (%)
    # ----------------------------------------------------------------
    availability = df_main["availability_pct"].mean()

    capex_total = (
        solar_capacity * capex_solar +
        wind_capacity * capex_wind +
        max_capacity_bess * capex_bess
    )

    # ----------------------------------------------------------------
    #  Create Results DataFrame (normalized)
    # ----------------------------------------------------------------
    df_results = pd.DataFrame({
        "parameter": [
            "market_sales_peak",
            "market_sales_offpeak",
            "cuf",
            "penalty",
            "availability",
            "capex"
        ],
        "unit": [
            "mwh",
            "mwh",
            "percent",
            "mwh",
            "percent",
            "rs_cr"
        ],
        "value": [
            market_sales_peak,
            market_sales_offpeak,
            cuf,
            penalty,
            availability,
            capex_total
        ]
    })

    return df_results


def generate_project_timeline_table(df_assumptions: pd.DataFrame):
    """
    Generate a dynamic project lifecycle table (construction–operations–loan)
    aligned with financial years.
    
    Key Behaviors:
    - First start_date = construction_start_date
    - Last end_date   = operation_end_date
    - days_in_year = total FY days (Apr–Mar), not truncated
    - Fractions computed as overlap / full FY days
    - Stops automatically once contract_year == defined limit (e.g., 25)
    """

    # ----------------------------------------------------------------
    #  User-defined (hardcoded for now)
    # ----------------------------------------------------------------
    construction_start = datetime(2024, 8, 1)
    construction_years = get_val(df_assumptions, "construction_years")
    construction_end = datetime(2026, 7, 31)
    operation_start = datetime(2026, 8, 1)
    contract_years = get_val(df_assumptions, "contract_years")
    operation_end = datetime(2051, 7, 31)
    loan_start = datetime(2027, 8, 1)
    loan_end = datetime(2045, 7, 31)

    # ----------------------------------------------------------------
    #  Generate financial-year periods (Apr → Mar)
    # ----------------------------------------------------------------
    fy_starts, fy_ends = [], []
    fy_start = datetime(2024, 4, 1)
    while fy_start <= operation_end:
        fy_starts.append(fy_start)
        fy_ends.append(datetime(fy_start.year + 1, 3, 31))
        fy_start = datetime(fy_start.year + 1, 4, 1)

    df = pd.DataFrame({"start_date": fy_starts, "end_date": fy_ends})

    # ----------------------------------------------------------------
    #  Adjust first and last rows to match actual project dates
    # ----------------------------------------------------------------
    df.loc[0, "start_date"] = construction_start
    df.loc[df.index[-1], "end_date"] = operation_end
    df = df[df["start_date"] <= df["end_date"]]

    # ----------------------------------------------------------------
    #  Helper: overlap_days
    # ----------------------------------------------------------------
    def overlap_days(a_start, a_end, b_start, b_end):
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        if overlap_start <= overlap_end:
            return (overlap_end - overlap_start).days + 1
        return 0

    # ----------------------------------------------------------------
    #  Days in each FY (full year)
    # ----------------------------------------------------------------
    df["days_in_year"] = df["start_date"].apply(
        lambda d: (datetime(d.year + 1, 3, 31) - datetime(d.year, 4, 1)).days + 1
    )

    # ----------------------------------------------------------------
    #  Calculate overlaps for construction and operation
    # ----------------------------------------------------------------
    df["construction_days"] = df.apply(
        lambda x: overlap_days(x.start_date, x.end_date, construction_start, construction_end), axis=1
    )
    df["operations_days"] = df.apply(
        lambda x: overlap_days(x.start_date, x.end_date, operation_start, operation_end), axis=1
    )

    # ----------------------------------------------------------------
    #  Assign contract_years dynamically and STOP when 25 is reached
    # ----------------------------------------------------------------
    df["contract_year"] = 0
    start_idx_list = df.index[df["start_date"] >= operation_start.replace(month=4, day=1)]

    if len(start_idx_list) > 0:
        start_idx = int(start_idx_list[0])  # ensure integer index
        total_rows = len(df) - start_idx
        assign_years = int(min(int(contract_years), total_rows))  # ensure int for range()

        df.loc[start_idx:start_idx + assign_years - 1, "contract_year"] = range(1, assign_years + 1)

        # Stop after final contract year
        cutoff_idx = start_idx + assign_years - 1
        df = df.loc[:cutoff_idx]

    # ----------------------------------------------------------------
    #  Fractions (6 significant digits)
    # ----------------------------------------------------------------
    df["construction_fraction"] = (df["construction_days"] / df["days_in_year"]).round(4)
    df["operations_fraction"] = (df["operations_days"] / df["days_in_year"]).round(4)

    # ----------------------------------------------------------------
    #  Loan counter logic
    # ----------------------------------------------------------------
    df["loan_counter"] = df.apply(
        lambda x: 1 if (x.end_date >= loan_start and x.start_date <= loan_end) else 0, axis=1
    )

    # ----------------------------------------------------------------
    #  Normalize column names
    # ----------------------------------------------------------------
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    return df


# ===============================================================
#  BUILD FINANCIAL MODEL
# ===============================================================

def build_financial_model(
        df_main: pd.DataFrame,
        df_static: pd.DataFrame,
        df_results: pd.DataFrame,
        df_timeline: pd.DataFrame,
        df_assumptions: pd.DataFrame,
    ):
    
    quoted_tariff         = get_val(df_assumptions, "quoted_tariff")
    market_tariff_peak    = get_val(df_assumptions, "market_tariff_peak")
    market_tariff_offpeak = get_val(df_assumptions, "market_tariff_offpeak")
    debt_ratio            = get_val(df_assumptions, "debt_ratio")
    loan_tenure           = get_val(df_assumptions, "loan_tenure")

    # -----------------------------------------------------------
    # Step 1 | Normalize and Prepare Base DataFrames
    # -----------------------------------------------------------
    df_static = normalize_columns(df_static)
    df_results = normalize_columns(df_results)
    df_timeline = normalize_columns(df_timeline)
    
    # ensure static date columns are datetime for exact matching
    df_static["start_date"] = pd.to_datetime(df_static["start_date"], dayfirst=True, errors="coerce")
    df_static["end_date"]   = pd.to_datetime(df_static["end_date"], dayfirst=True, errors="coerce")


    df = df_timeline.copy()[["start_date", "end_date", "operations_fraction", "loan_counter"]]

    # helper → get parameter value from df_results
    def get_value(param_name: str) -> float:
        s = df_results.loc[df_results["parameter"].str.lower() == param_name.lower(), "value"]
        return float(s.iloc[0]) if not s.empty else 0.0

    # helper → get static metric value for specific period
    def get_static(metric: str, s_date, e_date) -> float:
        s_date = pd.to_datetime(s_date).date()
        e_date = pd.to_datetime(e_date).date()

        s = df_static[
            (df_static["metrics"] == metric)
            & (pd.to_datetime(df_static["start_date"]).dt.date == s_date)
            & (pd.to_datetime(df_static["end_date"]).dt.date == e_date)
        ]
        return float(s["value"].iloc[0]) if not s.empty else 0.0


    # -----------------------------------------------------------
    # Step 2 | Extract Inputs from Result Data
    # -----------------------------------------------------------
    if "total_discom_demand_met" not in df_main.columns:
        raise KeyError("Critical column 'total_discom_demand_met' not found in df_main. Cannot proceed with financial model.")
    discom_demand = df_main["total_discom_demand_met"].sum()
    log(f"Total Discom Demand Met (MWh): {discom_demand}")
    
    market_sales_peak = get_value("market_sales_peak")
    market_sales_offpeak = get_value("market_sales_offpeak")
    penalty_val = get_value("penalty")
    capex_total = get_value("capex")

    # -----------------------------------------------------------
    # Step 3 | Calculated KPIs (Sales, Penalty, Revenue)
    # -----------------------------------------------------------
    df["sale_units_discom"] = discom_demand * 1000 * df["operations_fraction"]
    df["sale_units_merchant_peak"] = market_sales_peak * 1000 * df["operations_fraction"]
    df["sale_units_merchant_offpeak"] = market_sales_offpeak * 1000 * df["operations_fraction"]

    # Penalty (negative value since it’s a deduction)
    df["penalty"] = -penalty_val * 1000 * quoted_tariff * 1.5 / 1e7 * df["operations_fraction"]

    # Revenue (Rs Cr)
    df["revenue"] = (
        (df["sale_units_discom"] * quoted_tariff)
        + (df["sale_units_merchant_peak"] * market_tariff_peak)
        + (df["sale_units_merchant_offpeak"] * market_tariff_offpeak)
    ) / 1e7
    
    
    # -----------------------------------------------------------
    # Step 4 | Bring in Static Metrics (from df_static)
    # -----------------------------------------------------------
    static_metrics = [
        "o_m_expenses",
        "depreciation",
        "interest_on_term_loan",
        "interest_on_working_capital",
        "tax",
        "change_in_working_capital"
    ]

    for metric in static_metrics:
        df[metric] = [
            get_static(metric, s, e)
            for s, e in zip(df["start_date"], df["end_date"])
        ]

    # -----------------------------------------------------------
    # Step 5 | P&L Computation
    # -----------------------------------------------------------
    df["ebitda"] = df["revenue"] - (df["o_m_expenses"] + df["penalty"])
    df["interest_total"] = df["interest_on_term_loan"] + df["interest_on_working_capital"]
    df["pbt"] = df["ebitda"] - df["depreciation"] - df["interest_total"]
    df["pat"] = df["pbt"] - df["tax"]

    # -----------------------------------------------------------
    # Step 6 | Cash Flow Computation
    # -----------------------------------------------------------
    df["capex"] = 0.0
    df.loc[df.index[0], "capex"] = capex_total  # apply only to first period

    total_debt = capex_total * debt_ratio
    df["loan_inflow"] = df["capex"] * debt_ratio  # inflow 
    df["loan_repayment"] = (total_debt / loan_tenure) * df["loan_counter"]
    df["salvage_value"] = 0.0

    df["fcfe"] = (
        df["pat"]
        + df["depreciation"]
        - df["capex"]
        - df["change_in_working_capital"]
        + df["loan_inflow"]
        - df["loan_repayment"]
        + df["salvage_value"]
    )

    # -----------------------------------------------------------
    # Step 7 | Equity IRR (computed separately)
    # -----------------------------------------------------------
    try:
        equity_irr = round(xirr(df["end_date"], df["fcfe"]) * 100, 4)
    except Exception as e:
        print(f"[ERROR] XIRR computation failed: {e}")
        equity_irr = np.nan

    # -----------------------------------------------------------
    # Step 8 | Finalize Output
    # -----------------------------------------------------------
    col_order = [
        "start_date", "end_date", "operations_fraction", "loan_counter",
        "sale_units_discom", "sale_units_merchant_peak", "sale_units_merchant_offpeak",
        "penalty", "revenue",
        "o_m_expenses", "interest_on_term_loan", "interest_on_working_capital",
        "tax", "ebitda", "interest_total", "pbt", "pat", "depreciation",
        "capex", "change_in_working_capital", "loan_inflow", "loan_repayment",
        "salvage_value", "fcfe"
    ]

    df = df[[c for c in col_order if c in df.columns]]

    return df, equity_irr



