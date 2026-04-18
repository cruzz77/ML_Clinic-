from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import risk_ingestor, rag_retriever, llm_reasoner, report_builder, critic_node

def create_agent_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("risk_ingestor", risk_ingestor)
    workflow.add_node("rag_retriever", rag_retriever)
    workflow.add_node("llm_reasoner", llm_reasoner)
    workflow.add_node("report_builder", report_builder)
    workflow.add_node("critic", critic_node)
    
    # Define Edges (Linear pipeline)
    workflow.set_entry_point("risk_ingestor")
    workflow.add_edge("risk_ingestor", "rag_retriever")
    workflow.add_edge("rag_retriever", "llm_reasoner")
    workflow.add_edge("llm_reasoner", "report_builder")
    workflow.add_edge("report_builder", "critic")
    workflow.add_edge("critic", END)
    
    return workflow.compile()

def run_agent(patient_row: dict):
    """Entry point to run the agent for a single patient."""
    graph = create_agent_graph()
    
    initial_state = {
        "patient_row": patient_row,
        "prediction": {},
        "rag_context": None,
        "risk_profile": None,
        "llm_raw_output": None,
        "report": None,
        "error": None,
        "intervention_log": []
    }
    
    return graph.invoke(initial_state)
