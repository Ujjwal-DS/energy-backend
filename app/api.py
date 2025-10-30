# app/api.py
# ---------------------------------------------------------
# FastAPI wrapper for your Energy Model backend
# ---------------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import re

from app.schemas import RunRequest, RunResponse
from app.config import (
    LOAD_PROFILE_CSV, ASSUMPTIONS_CSV, ASSUMPTIONS_CAPEX_CSV, STATIC_NORM_CSV,
    OPT_ASSUMPTIONS_CSV, KEY_MAPPING_CSV,
    OUTPUT_BASE_DIR, ALLOWED_ORIGINS,
    FDRE_REFERENCE_DIR, OUTPUT_BASE_DIR, TENDER_VARIATION_MAP, DEFAULT_PEAK_HRS
)
from helpers import log, normalize_text
from preprocessing import preprocess_inputs
from logic import (
    generate_main_dataframe,
    generate_results_summary,
    generate_project_timeline_table,
    build_financial_model
)
from postprocessing import run_postprocessing
from utils import generate_dummy_outputs


# ---------------------------------------------------------
# 1️⃣ Initialize FastAPI App
# ---------------------------------------------------------
app = FastAPI(title="Energy Model API", version="1.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUT_BASE_DIR), name="files")


# ---------------------------------------------------------
# 2️⃣ Load Base Data (read once)
# ---------------------------------------------------------
DF_LOAD_PROFILE = pd.read_csv(LOAD_PROFILE_CSV)
DF_STATIC_NORM = pd.read_csv(STATIC_NORM_CSV)
DF_ASSUMPTIONS_BASE = pd.read_csv(ASSUMPTIONS_CSV)
DF_ASSUMPTIONS_CAPEX = pd.read_csv(ASSUMPTIONS_CAPEX_CSV)
DF_OPT_ASSUMPTIONS = pd.read_csv(OPT_ASSUMPTIONS_CSV)
DF_KEY_MAP = pd.read_csv(KEY_MAPPING_CSV)


# ---------------------------------------------------------
# 3️⃣ Helpers
# ---------------------------------------------------------
def norm_label(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00A0", " ").replace("\u200b", "")
    s = " ".join(s.split())
    return s.strip().lower()

def coerce_number(x):
    try:
        return float(x)
    except Exception:
        return x

def get_nested_anycase(d, path: str):
    """Case-insensitive nested getter for dotted paths."""
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict):
            return None
        match = next((k for k in cur.keys() if k.lower() == p.lower()), None)
        if match is None:
            return None
        cur = cur[match]
    return cur

def get_seed_from_opt_list(opt_list, label_text):
    if not isinstance(opt_list, list):
        return None
    want = norm_label(label_text)
    for item in opt_list:
        if norm_label(item.get("label", "")) == want:
            return coerce_number(item.get("seed"))
    return None

def get_capex_value(capex_obj, label_text):
    if not isinstance(capex_obj, dict):
        return None
    want = norm_label(label_text)
    for _, lst in capex_obj.items():
        if not isinstance(lst, list):
            continue
        for row in lst:
            if norm_label(row.get("label", "")) == want:
                return coerce_number(row.get("value"))
    return None


# ---------------------------------------------------------
# 4️⃣ Overrides via mapping (+ robust fallback alias map)
# ---------------------------------------------------------
def _apply_overrides_with_mapping(df_ass, df_opt, df_capex, payload, df_map):
    # normalize columns
    df_ass.columns = [c.strip().lower() for c in df_ass.columns]
    df_opt.columns = [c.strip().lower() for c in df_opt.columns]
    df_capex.columns = [c.strip().lower() for c in df_capex.columns]
    df_map.columns = [c.strip().lower() for c in df_map.columns]

    applied = 0

    # Stage 1: strict CSV mapping
    for _, r in df_map.iterrows():
        stype  = norm_label(r.get("source_type", ""))
        fe_key = str(r.get("fe_key", "")).strip()
        be_tbl = norm_label(r.get("be_table", ""))
        be_key = str(r.get("be_key", "")).strip()
        if not (stype and fe_key and be_tbl and be_key):
            continue

        val = None
        if stype == "path":
            val = get_nested_anycase(payload, fe_key)
        elif stype == "opt_label":
            val = get_seed_from_opt_list(payload.get("optimizationParams"), fe_key)
        elif stype == "capex_label":
            val = get_capex_value(payload.get("capex"), fe_key)

        val = coerce_number(val)
        if val is None:
            continue

        if be_tbl == "optimization_assumptions":
            mask = (df_opt["assumptions"] == be_key)
            if mask.any():
                df_opt.loc[mask, "value"] = val
                log(f"[MAP→OPT] {fe_key} → {be_key} = {val}")
                applied += 1
        elif be_tbl == "assumptions":
            mask = (df_ass["assumptions"] == be_key)
            if mask.any():
                df_ass.loc[mask, "value"] = val
                log(f"[MAP→BASE] {fe_key} → {be_key} = {val}")
                applied += 1
        elif be_tbl == "assumptions_capex":
            if "capex" in df_capex.columns:
                mask = (df_capex["capex"] == be_key)
                if mask.any():
                    df_capex.loc[mask, "value"] = val
                    log(f"[MAP→CAPEX] {fe_key} → {be_key} = {val}")
                    applied += 1

    if applied == 0:
        log("[MAP] No mapped overrides applied (check key_mapping.csv / hidden spaces).")

    # Stage 1.5: fallback alias map (covers finance/operational/optimization)
    alias_all = {
        # Finance
        "financeparams.debt": "debt_ratio",
        "financeparams.discountrate": "discount_rate",
        "financeparams.loantenor": "loan_tenure",
        # Operational
        "operationalparams.peakhours": "peak_hrs",
        # Optimization
        "solar capacity (mw)": "solar_capacity",
        "wind capacity (mw)": "wind_capacity",
        "battery storage (mwh)": "max_capacity_bess",
        "round trip efficiency (%)": "round_trip_eff_bess",
        "auxiliary consumption (%)": "aux_consumption",
        "dtl loss (%)": "dtl_losses",
        "quoted tariff": "quoted_tariff",
        "market tariff (peak)": "market_tariff_peak",
        "market tariff (off-peak)": "market_tariff_offpeak",
        "bid capacity (peak) (mw)": "bid_capacity_peak",
        "bid capacity (off peak) (mw)": "bid_capacity_offpeak",
    }

    fa_hits = 0
    for fe_key, be_key in alias_all.items():
        # try direct path value first
        val = get_nested_anycase(payload, fe_key)
        # then fall back to optimization labels (seed)
        if val is None:
            val = get_seed_from_opt_list(payload.get("optimizationParams"), fe_key)
        val = coerce_number(val)
        if val is None:
            continue

        # write to the correct table
        if be_key in df_opt["assumptions"].values:
            df_opt.loc[df_opt["assumptions"] == be_key, "value"] = val
            log(f"[FALLBACK→OPT] {fe_key} → {be_key} = {val}")
            fa_hits += 1
        elif be_key in df_ass["assumptions"].values:
            df_ass.loc[df_ass["assumptions"] == be_key, "value"] = val
            log(f"[FALLBACK→BASE] {fe_key} → {be_key} = {val}")
            fa_hits += 1
        elif "capex" in df_capex.columns and be_key in df_capex["capex"].values:
            df_capex.loc[df_capex["capex"] == be_key, "value"] = val
            log(f"[FALLBACK→CAPEX] {fe_key} → {be_key} = {val}")
            fa_hits += 1

    if fa_hits:
        log(f"[FALLBACK] {fa_hits} extra overrides applied via alias map.")

    # Stage 2: super-conservative top-level direct match (unchanged)
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                continue
            vv = coerce_number(v)
            if vv is None:
                continue
            key = str(k).strip()
            if (df_opt["assumptions"] == key).any():
                df_opt.loc[df_opt["assumptions"] == key, "value"] = vv
                log(f"[DIRECT→OPT] {key} = {vv}")
            elif (df_ass["assumptions"] == key).any():
                df_ass.loc[df_ass["assumptions"] == key, "value"] = vv
                log(f"[DIRECT→BASE] {key} = {vv}")
            elif "capex" in df_capex.columns and (df_capex["capex"] == key).any():
                df_capex.loc[df_capex["capex"] == key, "value"] = vv
                log(f"[DIRECT→CAPEX] {key} = {vv}")
                
    # ---- Stage 2.5: Direct CAPEX nested override ----
    capex_payload = payload.get("capex", {})
    if isinstance(capex_payload, dict):
        for group_name, items in capex_payload.items():  # e.g. "Solar", "Wind", ...
            if not isinstance(items, list):
                continue
            for row in items:
                label = str(row.get("label", "")).strip()
                value = coerce_number(row.get("value"))
                if not label or value is None:
                    continue
                # Match against df_capex['capex'] values
                if "capex" in df_capex.columns and (df_capex["capex"] == label).any():
                    df_capex.loc[df_capex["capex"] == label, "value"] = value
                    log(f"[DIRECT→CAPEX:NESTED] {group_name} → {label} = {value}")


    return df_ass, df_opt, df_capex


# ---------------------------------------------------------
# 5️⃣ Health + Defaults
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/defaults")
def get_defaults():
    log("Fetching default assumptions...")

    def norm(df, key_col):
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        df[key_col] = df[key_col].apply(normalize_text)
        return {r[key_col]: r["value"] for _, r in df.iterrows()}

    base = norm(DF_ASSUMPTIONS_BASE, "assumptions")
    opt  = norm(DF_OPT_ASSUMPTIONS, "assumptions")
    cap  = norm(DF_ASSUMPTIONS_CAPEX, "capex")

    merged = {**base, **opt, **cap}
    log("Defaults ready.")
    return {"assumptions": merged}


# ---------------------------------------------------------
# 6️⃣ Main Endpoint
# ---------------------------------------------------------
@app.post("/run", response_model=RunResponse)
def run_model(req: RunRequest):
    tender = req.tender_type.strip().lower()
    log(f"Request tender={req.tender_type}")

    if tender == "fdre assured peak model":
        log("Running full backend pipeline...")

        df_ass = DF_ASSUMPTIONS_BASE.copy()
        df_opt = DF_OPT_ASSUMPTIONS.copy()
        df_cap = DF_ASSUMPTIONS_CAPEX.copy()

        # Apply all overrides (incl. alias fallbacks)
        df_ass, df_opt, df_cap = _apply_overrides_with_mapping(
            df_ass, df_opt, df_cap, req.payload, DF_KEY_MAP
        )
        log("Overrides applied successfully.")

        # ---- Peak hours handling (global default + FE override)
        peak_hrs = None
        if isinstance(req.payload, dict):
            op = req.payload.get("operationalParams") or req.payload.get("operationalparams")
            if isinstance(op, dict):
                ph_val = op.get("peakHours") or op.get("peakhours")
                if isinstance(ph_val, str):
                    peak_hrs = [int(x.strip()) for x in ph_val.split(",") if x.strip()]
                elif isinstance(ph_val, list):
                    # list of numbers/strings
                    tmp = []
                    for x in ph_val:
                        try:
                            tmp.append(int(str(x).strip()))
                        except Exception:
                            pass
                    peak_hrs = tmp if tmp else None

        if not peak_hrs:
            peak_hrs = DEFAULT_PEAK_HRS
        log(f"[API] using peak_hrs = {peak_hrs}")

        # Merge optimization assumptions into base assumptions
        df_merged = pd.concat([df_ass, df_opt], ignore_index=True)
        log(f"Merged assumptions shape: {df_merged.shape}")

        # Preprocess
        df_load, df_ass_clean, df_cap_clean = preprocess_inputs(
            DF_LOAD_PROFILE.copy(), df_merged, df_cap
        )
        log("Preprocessing completed.")

        # Core model (NOTE: generate_main_dataframe must accept peak_hrs_override)
        df_main = generate_main_dataframe(df_load, df_ass_clean, peak_hrs=peak_hrs)

        df_penalty = (
            df_main.groupby("month", as_index=False)
            .agg({"shortfall": "sum"})
            .rename(columns={"shortfall": "penalty"})
        )
        df_results = generate_results_summary(df_main, df_ass_clean, df_cap_clean, df_penalty)
        df_timeline = generate_project_timeline_table(df_ass_clean)
        fin_df, eq_irr = build_financial_model(
            df_main=df_main,
            df_static=DF_STATIC_NORM.copy(),
            df_results=df_results,
            df_timeline=df_timeline,
            df_assumptions=df_ass_clean
        )
        log("Core model executed successfully.")

        # Post-processing
        post = run_postprocessing(df_main, fin_df, eq_irr, df_ass_clean)
        log("Post-processing done.")

        outputs = {
            "capacity_mix": post["df_capacity_mix"].to_dict("records"),
            "re_energy_profile": post["df_re_energy_profile"].to_dict("records"),
            "bess_schedule": post["df_bess_schedule"].to_dict("records"),
            "project_cashflows": post["df_project_cashflows"].to_dict("records"),
            "sensitivity_analysis": post["df_sensitivity_analysis"].to_dict("records"),
        }
        metrics = post["metrics"]

        log(f"FDRE run completed successfully. Equity IRR = {eq_irr:.2f}%")
        return {"metrics": metrics, "outputs": outputs, "message": "ok"}

    # Dummy for other tenders
    variation = TENDER_VARIATION_MAP.get("_default", 0.05)
    out_dir = os.path.join(OUTPUT_BASE_DIR, req.tender_type.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    log(f"Generating dummy outputs for {req.tender_type} (±{variation*100:.1f}%)")
    dummy_outputs, dummy_metrics = generate_dummy_outputs(
        reference_dir=str(FDRE_REFERENCE_DIR),
        output_dir=out_dir,
        variation=variation
    )
    outputs = {k: v.to_dict("records") for k, v in dummy_outputs.items()}
    log(f"Dummy outputs created for {req.tender_type}.")
    return {"metrics": dummy_metrics, "outputs": outputs, "message": "dummy run ok"}
