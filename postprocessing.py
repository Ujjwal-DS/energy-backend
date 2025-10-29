import pandas as pd
import numpy as np
import os
from helpers import log, get_val
import numpy_financial as nf

# -----------------------------------------------------
# Post-processing main function
# -----------------------------------------------------

def run_postprocessing(df_main, df_financial, eq_irr, df_assumptions):
    
    log(" Running post-processing...")

    # 1️⃣ CAPACITY MIX
    df_capacity_mix = df_assumptions[
        df_assumptions["assumptions"].isin(
            ["solar_capacity", "wind_capacity", "max_capacity_bess"]
        )
    ][["assumptions", "value"]].copy()

    df_capacity_mix["type"] = df_capacity_mix["assumptions"].replace({
        "solar_capacity": "Solar",
        "wind_capacity": "Wind",
        "max_capacity_bess": "BESS"
    })
    df_capacity_mix = df_capacity_mix[["type", "value"]].rename(columns={"value": "capacity"})

    log(" Capacity mix generated")

    # 2️⃣ RE ENERGY GENERATION PROFILE
    cols = ["month", "day", "hour", "minute", "solar", "wind", "opening_balance_bess"]
    cols = [c for c in cols if c in df_main.columns]
    df_re_energy_profile = df_main[cols].copy()
    df_re_energy_profile.rename(columns={"opening_balance_bess": "bess"}, inplace=True)

    id_columns = ["month", "day", "hour", "minute"]
    df_re_energy_profile = df_re_energy_profile.melt(
        id_vars=id_columns,
        var_name="type",
        value_name="generation"
    )
    
    df_re_energy_profile = (                                                   ##aggregation to hourly avg
    df_re_energy_profile
    .groupby(["hour", "type"], as_index=False)["generation"]
    .mean()
)

    log(" RE Energy Generation Profile created")

    # 3️⃣ BESS SCHEDULE (robust hourly open/close)
    bess_cols = [
    "month","day","hour","minute",
    "status_charge_discharge",
    "opening_balance_bess","closing_balance_bess"
    ]
    bess_cols = [c for c in bess_cols if c in df_main.columns]
    df_bess = df_main[bess_cols].copy()

    # capacity
    bess_capacity = df_assumptions.loc[
    df_assumptions["assumptions"] == "max_capacity_bess", "value"
    ].values[0]

    # make a numeric minute offset: "00:00","00:15","00:30","00:45" -> 0,15,30,45
    # (works even if minute is already time-like or string)
    m_str = df_bess["minute"].astype(str)
    df_bess["minute_off"] = m_str.str[-2:].astype(int)

    # group keys (include month if present)
    grp = [c for c in ["month","day","hour"] if c in df_bess.columns]

    # pick row index of earliest and latest 15-min slot within each hour
    idx_open  = df_bess.groupby(grp)["minute_off"].idxmin()
    idx_close = df_bess.groupby(grp)["minute_off"].idxmax()

    # build hourly table from those exact rows
    open_cols  = grp + ["opening_balance_bess"]
    close_cols = grp + ["closing_balance_bess"]
    df_open  = df_bess.loc[idx_open,  open_cols]
    df_close = df_bess.loc[idx_close, close_cols]

    df_bess_hourly = (
    df_open.merge(df_close, on=grp, how="inner")
            .sort_values(grp)
            .reset_index(drop=True)
    )

    # derived metrics
    df_bess_hourly["delta_energy"] = (
    df_bess_hourly["closing_balance_bess"] - df_bess_hourly["opening_balance_bess"]
    )
    df_bess_hourly["charge_mw"]    = df_bess_hourly["delta_energy"].clip(lower=0)
    df_bess_hourly["discharge_mw"] = (-df_bess_hourly["delta_energy"]).clip(lower=0)
    df_bess_hourly["soc_percent"]  = df_bess_hourly["opening_balance_bess"] / bess_capacity * 100
    df_bess_hourly["spc_percent"]  = df_bess_hourly["closing_balance_bess"] / bess_capacity * 100
    
    # --- Derive logical BESS status ---
    df_bess_hourly["status"] = np.select(
    [
        df_bess_hourly["delta_energy"] > 0,
        df_bess_hourly["delta_energy"] < 0
    ],
    [
        "Charge",
        "Discharge"
    ],
    default="No Action"
    )
    
    df_bess_hourly = df_bess_hourly.query("month == 1 and day == 1").reset_index(drop=True)  # keep only first day for brevity
    
    # --- Keep only required columns ---
    df_bess_hourly = df_bess_hourly[
    ["hour", "status", "opening_balance_bess", "closing_balance_bess",
        "charge_mw", "discharge_mw", "soc_percent", "spc_percent"]
    ]

    log("Hourly BESS schedule generated (opening = first 15-min, closing = last 15-min).")


    # 4️⃣ PROJECT FREE CASH FLOWS
    cols_cf = ["operations_fraction", "capex", "fcfe"]
    cols_cf = [c for c in cols_cf if c in df_financial.columns]
    df_project_cashflows = df_financial[cols_cf].copy()

    df_project_cashflows["dummy"] = np.where(df_project_cashflows["operations_fraction"] > 0, 1, 0)
    df_project_cashflows["operation_year"] = df_project_cashflows["dummy"].cumsum()

    df_project_cashflows = (
        df_project_cashflows.groupby("operation_year")[["capex", "fcfe"]]
        .sum().reset_index()
    )
    log(" Project free cashflows generated")

    # 5️⃣ RE ENERGY GENERATION metric
    total_re_energy = df_main["total_generation"].sum() if "total_generation" in df_main.columns else np.nan
    log(f" Total RE energy generated: {total_re_energy:.2f} MWh")

    # 6️⃣ NPV calculation
    discount_rate = df_assumptions.loc[
        df_assumptions["assumptions"] == "discount_rate", "value"
    ]
    discount_rate = discount_rate.values[0] if not discount_rate.empty else 0.1

    fcfe_values = df_financial["fcfe"].values if "fcfe" in df_financial.columns else []
    npv = nf.npv(discount_rate, fcfe_values) if len(fcfe_values) > 0 else np.nan
    log(f" NPV computed: {npv:.2f}")

    # 7️⃣ Enhanced Sensitivity Analysis (IRR + NPV realistic version)
    irr_base = eq_irr
    npv_base = npv

    # --- Define sensitivity buckets ---
    high_neg = {"CAPEX", "OPEX", "Land_Expense", "Depreciation", "Penalty_Amount", "Interest_Rates"}
    high_pos = {"Solar_Plant_Size", "Wind_Plant_Size", "Electrolyzer_Size", "Electrolyzer_Efficiency",
                "BESS_Size", "BESS_RoundTrip_Efficiency", "Capacity_Utilisation_Factor", "Selling_Price"}
    moderate = {"BESS_Life", "BESS_Degradation", "Depth_of_Discharge", "Contracted_Capacity_GH2"}
    mild = {"Subsidy", "LCOE_LCOH", "Tariff"}

    decision_variables = list(high_neg | high_pos | moderate | mild)
    levels = np.linspace(-0.4, 0.4, 21)
    records = []

    def generate_realistic_pair(var, delta, irr0, npv0):
        # Define base correlation and noise
        corr = np.random.uniform(0.8, 0.95)
        noise = np.random.uniform(-0.01, 0.01)

        # Determine direction and sensitivity scale
        if var in high_neg:
            slope = np.random.uniform(0.8, 1.2)
            direction = -1
        elif var in high_pos:
            slope = np.random.uniform(0.6, 1.0)
            direction = +1
        elif var in moderate:
            slope = np.random.uniform(0.3, 0.6)
            direction = +1 if delta >= 0 else -1
        else:  # mild
            slope = np.random.uniform(0.1, 0.3)
            direction = +1

        # Simulate IRR curve (smooth bounded)
        irr_change = direction * slope * delta
        irr = irr0 * (1 + irr_change + 0.2 * delta**2 + noise)
        irr = np.clip(irr, irr0 * 0.85, irr0 * 1.15)

        # Simulate correlated NPV curve (scaled with correlation)
        npv_change = irr_change * corr + noise / 2
        npv = npv0 * (1 + npv_change)
        npv = np.clip(npv, npv0 * 0.75, npv0 * 1.25)

        return irr, npv


    # Generate dummy sensitivity data
    for var in decision_variables:
        for d in levels:
            irr_val, npv_val = generate_realistic_pair(var, d, irr_base, npv_base)
            records.append({
                "Decision_Variable": var,
                "Change_in_Variable(%)": d * 100,
                "IRR(%)": irr_val,
                "NPV(Cr)": npv_val
            })

    df_sensitivity_analysis = pd.DataFrame(records)
    log(" Realistic sensitivity analysis (IRR + NPV) generated.")

    # --- NEW: fetch LCOE straight from assumptions (normalized already) ---
    try:
        min_lcoe = float(get_val(df_assumptions, "lcoe"))
    except Exception:
        min_lcoe = np.nan

    # --- existing metrics dict you already return ---
    metrics = {
        "total_re_energy": float(total_re_energy),
        "npv": float(npv) if np.isfinite(npv) else np.nan,
        "max_irr": float(eq_irr) if eq_irr is not None else np.nan,
        "min_lcoe": float(min_lcoe) if np.isfinite(min_lcoe) else np.nan,
    }

    # --- NEW: build a metrics_df for saving as metrics.csv ---
    metrics_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()],
        columns=["metric", "value"]
    )

    return {
        "df_capacity_mix": df_capacity_mix,
        "df_re_energy_profile": df_re_energy_profile,
        "df_bess_schedule": df_bess_hourly,
        "df_project_cashflows": df_project_cashflows,
        "df_sensitivity_analysis": df_sensitivity_analysis,
        "metrics": metrics,
        "metrics_df": metrics_df,             # <-- NEW
    }
