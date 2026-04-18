from pydantic import BaseModel, Field
from typing import List, Literal

class Intervention(BaseModel):
    action: str = Field(..., description="Specific action for the care team to take")
    rationale: str = Field(..., description="Clinical or operational logic for this intervention")
    priority: Literal["high", "medium", "low"]
    effort: str = Field(..., description="Estimated effort to implement (e.g., '10 min call', 'Home visit')")

class CareCoordinationReport(BaseModel):
    appointment_risk_summary: str = Field(..., description="Unified summary of why this appointment is at risk")
    contributing_factors: List[str] = Field(..., description="Top features or factors driving the risk (max 6)", max_length=6)
    intervention_strategies: List[Intervention] = Field(..., description="Actionable interventions (max 5)", max_length=5)
    sources: List[str] = Field(..., description="Citations from clinical guidelines used for interventions")
    confidence: Literal["high", "medium", "low"]
    disclaimers: List[str] = Field(..., description="Operational and ethical disclaimers")
