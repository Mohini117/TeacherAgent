import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field 
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
import operator
import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langchain_groq import ChatGroq
from langgraph.constants import Send

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="Gemma2-9b-It", temperature=0.7)

# Function to load local CSS
def local_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS file
local_css("styles.css")

# Schema for structured output
class Section(BaseModel):
    name: str = Field(description="📌  role name (e.g., Math, Physics, Chemistry, Science).")
    description: str = Field(description="Details of the subject query.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections representing teacher roles.")

# Augment the LLM with structured output schema
planner = llm.with_structured_output(Sections)

# Graph State
class State(TypedDict):
    topic: str
    sections: List[Dict]
    completed_sections: Annotated[List, operator.add]
    final_report: str

# Worker State
class WorkerState(TypedDict):
    section: Dict
    completed_sections: Annotated[List, operator.add]

# Node: Teacher agent explains the topic (generic teacher node creator)
def teacher_node(role: str):
    def teacher(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content=f"You are a {role} teacher. Provide a clear and comprehensive explanation using markdown formatting."),
            HumanMessage(content=f"Query details: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content], "section": state["section"]}
    return teacher

# Node: Feedback mechanism after teacher explanation
def student_feedback(state: WorkerState):
    # Display the teacher's explanation 
    teacher_response = state["completed_sections"][-1]   
    st.markdown("### 👩‍🏫 Teacher's Explanation:")
    st.markdown(teacher_response)
    
    # Provide a feedback input area for the student
    st.markdown("### Your Feedback:")
    user_feedback = st.text_area("Does the explanation satisfy your query? If not, provide your feedback below:")
    
    # Wait for the student to submit feedback
    if st.button("Submit Feedback"):
        if user_feedback.strip():
            # If feedback is provided, improve the answer accordingly
            improved = llm.invoke([
                SystemMessage(content="You are an AI teacher tasked with refining your explanation based on student feedback."),
                HumanMessage(content=f"Feedback: {user_feedback}\nPrevious Answer: {teacher_response}")
            ])
            st.markdown("### Improved Explanation:")
            st.markdown(improved.content)
            return {"completed_sections": [improved.content], "section": state["section"]}
        else:
            # If no feedback is provided, return the original answer
            return {"completed_sections": [teacher_response], "section": state["section"]}
    # If the button hasn't been clicked yet, remain at this node (this might require a re-run of the Streamlit app)
    st.info("Awaiting your feedback submission...")
    return {"completed_sections": [teacher_response], "section": state["section"]}

# Orchestrator node: assign teacher roles based on the query
def orchestrator(state: State):
    report_sections = planner.invoke([
        SystemMessage(content="Decide which teacher should be assigned to answer the following subject query. The available teachers are for Math, Physics, Chemistry, and Science. Also decide if a basic or advanced explanation is needed based on the query details."),
        HumanMessage(content=f"Subject query: {state['topic']}")
    ])
    if not report_sections or not report_sections.sections:
        raise ValueError("No sections were generated for the query.")
    sections_list = [section.model_dump() for section in report_sections.sections]
    return {"sections": sections_list}

# Synthesizer node: Combine teacher responses into a final answer
def synthesizer(state: State):
    final_report = "\n\n---\n\n".join(state["completed_sections"])
    return {"final_report": final_report}

# Build the workflow
def q_and_a_manager():
    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("math_teacher", teacher_node("Math"))
    builder.add_node("physics_teacher", teacher_node("Physics"))
    builder.add_node("chemistry_teacher", teacher_node("Chemistry"))
    builder.add_node("science_teacher", teacher_node("Science"))
    builder.add_node("student_feedback", student_feedback)
    builder.add_node("synthesizer", synthesizer)
    
    def assign_teacher(state: State):
        worker_mapping = {
            "Math": "math_teacher",
            "Physics": "physics_teacher",
            "Chemistry": "chemistry_teacher",
            "Science": "science_teacher"
        }
        return [Send(worker_mapping.get(s["name"], "science_teacher"), {"section": s}) for s in state["sections"]]
    
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_teacher, ["math_teacher", "physics_teacher", "chemistry_teacher", "science_teacher"])
    builder.add_edge("math_teacher", "student_feedback")
    builder.add_edge("physics_teacher", "student_feedback")
    builder.add_edge("chemistry_teacher", "student_feedback")
    builder.add_edge("science_teacher", "student_feedback")
    builder.add_edge("student_feedback", "synthesizer")
    builder.add_edge("synthesizer", END)
    
    return builder.compile()

# Initialize the agent
agent = q_and_a_manager()

# --- Streamlit Interface ---
def main():
    st.markdown("""
        <div class="main-header">
            <h1>Advanced Subject Teacher AI Agent</h1>
            <p class="subtitle">Your ultimate source for interactive learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar: Display the workflow diagram
    with st.sidebar:
        st.subheader("Workflow Diagram")
        try:
            graph_img = agent.get_graph().draw_mermaid_png()
            st.image(graph_img, caption="Workflow Diagram", use_container_width=True)
        except Exception as e:
            st.write("Could not display workflow diagram:", e)
    
    st.subheader("Chatbot")
    query = st.text_area("Ask a question about Math, Physics, Chemistry, or Science:")
    
    if st.button("Get Answer"):
        if query.strip():
            try:
                state = {
                    "topic": query,
                    "sections": [],
                    "completed_sections": [],
                    "final_report": ""
                }
                result = agent.invoke(state)
                # st.markdown("### 📌 Final Answer:")
                # st.markdown(result["final_report"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query before submitting.")

if __name__ == "__main__":
    main()
