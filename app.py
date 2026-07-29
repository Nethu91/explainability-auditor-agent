import streamlit as st
from agents.retriever_agent import retrieve_and_explain
from agents.critic_agent import critique_explanation

st.set_page_config(page_title="Explainability Auditor", page_icon="🔍")

st.title("🔍 Explainability Auditor")
st.write("Agentic AI system that explains student financial-risk predictions, grounded in XAI literature.")

with st.form("risk_form"):
    student_id = st.text_input("Student ID", value="S1023")
    risk_level = st.selectbox("Risk Level", ["Low", "Medium", "High"])
    factors_input = st.text_area(
        "Contributing Factors (comma-separated)",
        value="low monthly income, high dependency ratio, delayed fee payments"
    )
    submitted = st.form_submit_button("Generate Explanation")

if submitted:
    factors = [f.strip() for f in factors_input.split(",")]
    risk_flag = {"student_id": student_id, "risk_level": risk_level, "factors": factors}

    try:
        with st.spinner("Agent 1: Retrieving literature and drafting explanation..."):
            agent1_result = retrieve_and_explain(risk_flag)

        st.subheader("📝 Draft Explanation (Agent 1)")
        st.write(agent1_result["draft_explanation"])

        with st.spinner("Agent 2: Validating against literature..."):
            agent2_result = critique_explanation(agent1_result)

        st.subheader("✅ Validation Result (Agent 2)")
        st.write("**Approved:**", agent2_result["approved"])
        st.write("**Issues found:**")
        st.write(agent2_result["issues"])
        st.write("**Final Explanation:**")
        st.write(agent2_result["revised_explanation"])

        with st.expander("📚 Retrieved source chunks"):
            for i, chunk in enumerate(agent1_result["retrieved_context"]):
                st.text(f"Chunk {i+1}:\n{chunk[:300]}...")

    except Exception as e:
        st.error(f"Something went wrong while generating the explanation: {e}")
        st.info("Please check your API keys and try again.")