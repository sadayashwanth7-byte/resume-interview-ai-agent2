
import streamlit as st
from pypdf import PdfReader
from google import genai
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from io import BytesIO


# ---------------------------------
# Streamlit Configuration
# ---------------------------------

st.set_page_config(
    page_title="Resume & Interview AI Agent",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Resume & Interview Preparation AI Agent")

st.write(
    "Upload your resume, enter your target job description, "
    "and receive AI-powered resume feedback and interview preparation."
)


# ---------------------------------
# Gemini API Configuration
# ---------------------------------

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = ""

if not api_key:
    st.error(
        "Gemini API key is not configured. "
        "Add GEMINI_API_KEY under Streamlit App Settings → Secrets."
    )
    st.stop()


@st.cache_resource
def get_gemini_client(key):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# ---------------------------------
# Extract Resume PDF Text
# ---------------------------------

def extract_resume_text(uploaded_file):

    pdf = PdfReader(BytesIO(uploaded_file.getvalue()))

    pages = []

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


# ---------------------------------
# LangGraph State
# ---------------------------------

class ResumeState(TypedDict):
    resume_text: str
    job_description: str
    target_role: str
    report: str


# ---------------------------------
# Node 1: Prepare Resume Information
# ---------------------------------

def prepare_information(state: ResumeState):

    resume_text = " ".join(
        state["resume_text"].split()
    )

    job_description = state["job_description"].strip()

    target_role = state["target_role"].strip()

    return {
        "resume_text": resume_text[:18000],
        "job_description": job_description[:10000],
        "target_role": target_role or "Not specified"
    }


# ---------------------------------
# Node 2: Generate Career Report
# ---------------------------------

def generate_report(state: ResumeState):

    prompt = f"""
You are a professional Resume Review and Interview Preparation AI Agent.

Analyze the student's resume and provide practical career guidance.

TARGET JOB ROLE:
{state["target_role"]}

JOB DESCRIPTION:
{state["job_description"] or "No job description provided."}

RESUME TEXT:
{state["resume_text"]}

Create a report with these sections:

1. RESUME SUMMARY
Summarize the candidate's profile using information in the resume.

2. STRENGTHS
Identify the candidate's strongest skills, projects, and experiences.
Use only information supported by the resume.

3. AREAS TO IMPROVE
Identify missing information, weak bullet points, and presentation issues.

4. JOB MATCH AND SKILL GAPS
Compare the resume with the supplied job description.
If no job description is supplied, provide general recommendations.
Never claim that an unlisted skill is definitely absent; describe it as
not demonstrated in the supplied resume.

5. RESUME IMPROVEMENTS
Suggest stronger wording for up to three resume bullet points.
Do not invent employers, qualifications, achievements, metrics, or skills.
Clearly label any suggested content that requires confirmation.

6. INTERVIEW PREPARATION
Provide six relevant interview questions.
Include a suggested answer outline for each question.
Use the resume to personalize the answers, without inventing experience.

7. ACTION PLAN
Give five practical actions the candidate can take to improve the resume
and prepare for the target role.

Be specific, constructive, professional, and easy for a student to follow.
Use headings and bullet points.
Do not guarantee employment or interview selection.
"""

    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt,
        generation_config={
            "thinking_level": "minimal"
        }
    )

    return {
        "report": interaction.output_text
    }


# ---------------------------------
# Build LangGraph Workflow
# ---------------------------------

def build_agent():

    builder = StateGraph(ResumeState)

    builder.add_node(
        "prepare",
        prepare_information
    )

    builder.add_node(
        "analyze",
        generate_report
    )

    builder.add_edge(
        START,
        "prepare"
    )

    builder.add_edge(
        "prepare",
        "analyze"
    )

    builder.add_edge(
        "analyze",
        END
    )

    return builder.compile()


resume_agent = build_agent()


# ---------------------------------
# User Inputs
# ---------------------------------

st.subheader("📄 Step 1: Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume in PDF format",
    type=["pdf"]
)

st.subheader("🎯 Step 2: Select Your Career Goal")

target_role = st.text_input(
    "Target job role",
    placeholder="Example: Python Developer"
)

job_description = st.text_area(
    "Job description (optional)",
    placeholder="Paste the job description here for a more targeted analysis.",
    height=180
)


# ---------------------------------
# Run the AI Agent
# ---------------------------------

if st.button("🚀 Analyze Resume & Prepare Me", type="primary"):

    if uploaded_file is None:
        st.warning("Please upload your resume PDF first.")

    elif not target_role.strip():
        st.warning("Please enter your target job role.")

    else:

        try:

            with st.spinner("Reading your resume..."):

                resume_text = extract_resume_text(uploaded_file)

            if not resume_text.strip():

                st.error(
                    "No readable text was found in the PDF. "
                    "Please upload a text-based PDF rather than a scanned image."
                )

            else:

                with st.spinner(
                    "🤖 Analyzing your resume and preparing interview questions..."
                ):

                    result = resume_agent.invoke({
                        "resume_text": resume_text,
                        "job_description": job_description,
                        "target_role": target_role,
                        "report": ""
                    })

                st.success("✅ Resume analysis completed!")

                st.markdown(result["report"])

                st.download_button(
                    label="📥 Download Your Career Report",
                    data=result["report"],
                    file_name="resume_interview_report.txt",
                    mime="text/plain"
                )

        except Exception as error:

            st.error(
                "The agent encountered an error. "
                "Please try again. If it continues, check the Streamlit Cloud logs."
            )

            st.caption(f"Error details: {error}")


# ---------------------------------
# Footer
# ---------------------------------

st.divider()

st.caption(
    "Powered by Python, Streamlit, Gemini, and LangGraph. "
    "AI-generated suggestions should be reviewed before use."
)
