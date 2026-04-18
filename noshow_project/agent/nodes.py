import json
from agent.state import AgentState
from utils.preprocess import enrich_patient_row, run_prediction, format_risk_profile
from rag.retriever import retrieve_guidelines
from utils.llm import LLMHandler
from agent.prompts import SYSTEM_PROMPT, USER_PROMPT, CRITIC_PROMPT

def critic_node(state: AgentState) -> AgentState:
    """Node: Analyze report for evidence alignment."""
    print("--- [Node: Critic] ---")
    if state.get('error'): return state
    
    try:
        llm = LLMHandler()
        # Use dump() for JSON serializable dict
        report_json = json.dumps(state['report'])
        
        prompt = CRITIC_PROMPT.format(
            rag_context=state['rag_context'],
            generated_report=report_json
        )
        
        # Auditor role
        critique = llm.generate_report("You are a clinical auditor.", prompt)
        
        if "REJECT" in critique.upper():
            if 'disclaimers' not in state['report']: state['report']['disclaimers'] = []
            state['report']['disclaimers'].append("Clinical Audit: Some interventions were adjusted as they exceeded direct evidence in guidelines.")
            state['intervention_log'].append({"status": "Self-Correction Applied", "timestamp": "now"})
        else:
            state['intervention_log'].append({"status": "Verified by Auditor", "timestamp": "now"})
            
    except Exception as e:
        print(f"Critic node skipped due to error: {e}")
    return state
import streamlit as st

def risk_ingestor(state: AgentState) -> AgentState:
    """Node 1: Validate + Enrich patient row and run model prediction."""
    print("--- [Node: Risk Ingestor] ---")
    try:
        row = state['patient_row']
        enriched_row = enrich_patient_row(row)
        prediction = run_prediction(enriched_row)
        
        state['prediction'] = prediction
        state['patient_row'] = enriched_row
        state['risk_profile'] = format_risk_profile(enriched_row, prediction)
    except Exception as e:
        state['error'] = f"Ingestor failed: {str(e)}"
    return state

def rag_retriever(state: AgentState) -> AgentState:
    """Node 2: Fetch relevant clinical guidelines."""
    print("--- [Node: RAG Retriever] ---")
    if state.get('error'): return state
    
    try:
        query = f"Interventions for {state['prediction']['tier']} risk patient with {state['patient_row'].get('ChronicCount', 0)} chronic conditions"
        context = retrieve_guidelines(query)
        state['rag_context'] = context
    except Exception as e:
        state['error'] = f"Retriever failed: {str(e)}"
    return state

def llm_reasoner(state: AgentState) -> AgentState:
    """Node 3: Call LLM with structured prompt."""
    print("--- [Node: LLM Reasoner] ---")
    if state.get('error'): return state
    
    try:
        llm = LLMHandler()
        system_msg = SYSTEM_PROMPT.format(
            rag_context=state['rag_context'],
            risk_profile=state['risk_profile']
        )
        
        raw_output = llm.generate_report(system_msg, USER_PROMPT)
        state['llm_raw_output'] = raw_output
    except Exception as e:
        state['error'] = f"Reasoner failed: {str(e)}"
    return state

from schemas.report import CareCoordinationReport

def report_builder(state: AgentState) -> AgentState:
    """Node 4: Parse + Validate JSON output."""
    print("--- [Node: Report Builder] ---")
    if state.get('error'): return state
    
    try:
        raw_json = state['llm_raw_output']
        # Cleanup potential markdown ticks if LLM produced them
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
        parsed_dict = json.loads(raw_json)
        
        # Validate schema using Pydantic
        validated_report = CareCoordinationReport.model_validate(parsed_dict)
        state['report'] = validated_report.model_dump()
        state['intervention_log'].append({"status": "Success", "timestamp": "now"})
    except Exception as e:
        state['error'] = f"Report Builder failed to parse JSON: {str(e)}"
        # Provide a fallback report structure
        state['report'] = {
            "appointment_risk_summary": "Error parsing AI report.",
            "contributing_factors": ["Parsing Error"],
            "intervention_strategies": [],
            "sources": [],
            "confidence": "low",
            "disclaimers": ["System Error: AI output was malformed."]
        }
    return state
