# main.py
# -----------------------------------------------------------
# Entry-point: orchestrates all Energy model computations
# -----------------------------------------------------------

import os
import pandas as pd
from helpers import log
from preprocessing import preprocess_inputs
from logic import (
    generate_main_dataframe,
    generate_results_summary,
    generate_project_timeline_table,
    build_financial_model
)
from postprocessing import run_postprocessing
from utils import generate_dummy_outputs

tender_type = "FDRE Assured Peak Model"       #"FDRE Assured Peak Model"

if tender_type.lower() == "fdre assured peak model":
    
    log(f"\n---------------------------------Running full backend pipeline for {tender_type}--------------------------------------\n")
    
    # -----------------------------------------------------------
    # Step 1 | Read Input Data
    # -----------------------------------------------------------
    df_load_profile = pd.read_csv(r"D:\GH2\Energy\input\solar_wind_profile.csv")
    df_assumptions = pd.read_csv(r"D:\GH2\Energy\input\assumptions.csv")
    df_assumptions_capex = pd.read_csv(r"D:\GH2\Energy\input\assumptions_capex.csv")
    df_static_normalized = pd.read_csv(r"D:\GH2\Energy\input\df_static_normalized.csv")

    # -----------------------------------------------------------
    # Step 2 | Preprocess Inputs
    # -----------------------------------------------------------
    df_load_profile, df_assumption, df_assumptions_capex = preprocess_inputs(df_load_profile, df_assumptions, df_assumptions_capex)

    print(df_assumptions_capex)

    # -----------------------------------------------------------
    # Step 3 | Generate Main DF (BESS Simulation)
    # -----------------------------------------------------------
    df_main = generate_main_dataframe(df_load_profile, df_assumptions)
    log("df_main generated successfully")


    # -----------------------------------------------------------
    # Step 4 | Derive Monthly Shortfall (for penalty)
    # -----------------------------------------------------------
    df_monthly_shortfall = (
        df_main.groupby("month", as_index=False)
        .agg({"shortfall": "sum"})
        .rename(columns={"shortfall": "penalty"})
        .sort_values("month")
        .reset_index(drop=True)
    )
    log(" Monthly shortfall computed successfully")

    # -----------------------------------------------------------
    # Step 5 | Generate Results Summary
    # -----------------------------------------------------------
    df_results = generate_results_summary(df_main, df_assumptions, df_assumptions_capex, df_monthly_shortfall)
    log(f"Results summary created: {df_results.shape}")

    # -----------------------------------------------------------
    # Step 6 | Generate Project Timeline
    # -----------------------------------------------------------
    df_timeline = generate_project_timeline_table(df_assumptions)
    log(f"Project timeline generated: {df_timeline.shape}")

    # -----------------------------------------------------------
    # Step 7 | Build Financial Model
    # -----------------------------------------------------------
    financial_df, eq_irr = build_financial_model(
        df_main=df_main,
        df_static=df_static_normalized,
        df_results=df_results,
        df_timeline=df_timeline,
        df_assumptions=df_assumptions
    )

    log(f" Financial model built successfully. Equity IRR = {eq_irr:.2f}%")

    #-----------------------------------------------------------
    # Step 8 | Post-processing      
    # -----------------------------------------------------------

    post_results = run_postprocessing(df_main, financial_df, eq_irr, df_assumptions)

    df_capacity_mix          = post_results["df_capacity_mix"]
    df_re_energy_profile     = post_results["df_re_energy_profile"]
    df_bess_schedule         = post_results["df_bess_schedule"]
    df_project_cashflows     = post_results["df_project_cashflows"]
    df_sensitivity_analysis  = post_results["df_sensitivity_analysis"]
    total_re_energy          = post_results["metrics"]["total_re_energy"]
    npv                      = post_results["metrics"]["npv"]
    max_irr                  = eq_irr
    min_lcoe                 = df_assumptions.loc[df_assumptions["assumptions"] == "lcoe", "value"].values[0]
    metrics_df = post_results["metrics_df"] 
    
    log("Post-processing completed successfully.")

    # -----------------------------------------------------------
    # Step 9 | Save Outputs (optional)
    # -----------------------------------------------------------
    output_dir = fr"D:\GH2\Energy\output\{tender_type.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)  # creates folder if missing

    # Core outputs
    # financial_df.to_csv(f"{output_dir}\\financial_df.csv", index=False)
    # df_results.to_csv(f"{output_dir}\\results_summary.csv", index=False)
    # df_main.to_csv(f"{output_dir}\\main_dataframe.csv", index=False)

    # Post-processing outputs
    df_capacity_mix.to_csv(f"{output_dir}\\capacity_mix.csv", index=False)
    df_re_energy_profile.to_csv(f"{output_dir}\\re_energy_generation_profile.csv", index=False)
    df_bess_schedule.to_csv(f"{output_dir}\\bess_schedule.csv", index=False)
    df_project_cashflows.to_csv(f"{output_dir}\\project_free_cashflows.csv", index=False)
    df_sensitivity_analysis.to_csv(f"{output_dir}\\sensitivity_analysis.csv", index=False)
    metrics_df.to_csv(f"{output_dir}\\metrics.csv", index=False)

    log(" All outputs (core + post-processing) saved successfully")
    
else:
    
    log(f"--------------------------------------Genarating dummy output files for {tender_type}-----------------------------------------")
    
    reference_dir = r"D:\GH2\Energy\output\FDRE_Assured_Peak_Model"
    output_dir = fr"D:\GH2\Energy\output\{tender_type.replace(' ', '_')}"
    dummy_outputs, dummy_metrics = generate_dummy_outputs(reference_dir, output_dir)

    # Optionally, save the returned dummy metrics to CSV for consistency
    if dummy_metrics:
        metrics_df = pd.DataFrame(list(dummy_metrics.items()), columns=["metric", "value"])
        metrics_df.to_csv(f"{output_dir}\\metrics.csv", index=False)
        log(f"Dummy metrics.csv saved for tender: {tender_type}")

    log(f"Dummy outputs generated for tender type: {tender_type}")
