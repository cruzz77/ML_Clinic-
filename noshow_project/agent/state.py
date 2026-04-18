from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    patient_row:      dict           # raw row from dataset
    prediction:       dict           # {probability, tier, top_features}
    rag_context:      Optional[str]  # retrieved guideline chunks
    risk_profile:     Optional[str]  # formatted prompt string
    llm_raw_output:   Optional[str]  # raw LLM JSON string
    report:           Optional[dict] # parsed + validated report
    error:            Optional[str]  # error message if any
    intervention_log: List[dict]     # session history
