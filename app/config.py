# app/config.py
# ---------------------------------------------------------
# Central configuration for all file paths and global settings
# ---------------------------------------------------------

from pathlib import Path

# ---------------------------------------------------------
#  Base Directories
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent        # D:/GH2/Energy
INPUT_DIR = BASE_DIR / "input"                           # Input data directory
OUTPUT_BASE_DIR = BASE_DIR / "output"                    # All tender output subfolders live here


# ---------------------------------------------------------
#  Input File Paths
# ---------------------------------------------------------
LOAD_PROFILE_CSV        = INPUT_DIR / "solar_wind_profile.csv"
STATIC_NORM_CSV         = INPUT_DIR / "df_static_normalized.csv"
ASSUMPTIONS_CSV         = INPUT_DIR / "assumptions.csv"
ASSUMPTIONS_CAPEX_CSV   = INPUT_DIR / "assumptions_capex.csv"

# Optimization parameters and key mapping
OPT_ASSUMPTIONS_CSV     = INPUT_DIR / "optimization_assumptions.csv"
KEY_MAPPING_CSV         = INPUT_DIR / "key_mapping.csv"


# ---------------------------------------------------------
#  Reference Folder for Dummy Outputs
# ---------------------------------------------------------
FDRE_REFERENCE_DIR = OUTPUT_BASE_DIR / "FDRE_Assured_Peak_Model"


# ---------------------------------------------------------
#  API and Server Settings
# ---------------------------------------------------------
API_PORT = 8000
ALLOWED_ORIGINS = ["*"]  # Allow all — adjust later for production


# ---------------------------------------------------------
#  Variation Settings (±%) for Dummy Runs by Tender
# ---------------------------------------------------------
TENDER_VARIATION_MAP = {
    "gh2 model": 0.06,
    "fdre load following model": 0.05,
    "solar + bess": 0.07,
    "standalone bess": 0.08,

    # Default fallback if tender not listed
    "_default": 0.05,
}

# ----- Default operational parameters -----
DEFAULT_PEAK_HRS = [9, 10, 11, 12]  # hours (default if frontend doesn’t send)


# ---------------------------------------------------------
#  Derived / Utility Constants
# ---------------------------------------------------------
# Default output folder for current tender run (created dynamically)
OUTPUT_DIR = OUTPUT_BASE_DIR
