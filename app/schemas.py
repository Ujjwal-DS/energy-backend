# app/schemas.py
from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional

TenderLiteral = Literal[
    "FDRE Assured Peak Model",
    "GH2 Model",
    "FDRE Load Following Model",
    "Solar + Bess",
    "Standalone BESS"
]

class RunRequest(BaseModel):
    """
    What FE sends.
    - tender_type: which tender the user picked
    - payload: the entire JSON (nested) from FE (financeParams, capex, etc.)
    - output_level: optional, not used by backend logic, just for previews if needed
    """
    tender_type: TenderLiteral
    payload: Dict[str, Any] = {}
    output_level: Literal["summary", "detailed"] = "summary"


class RunResponse(BaseModel):
    """
    What BE returns.
    """
    metrics: Dict[str, float]        # e.g., total_re_energy, npv, max_irr, min_lcoe
    outputs: Dict[str, list]         # each DF as list[dict]
    message: str = "ok"
