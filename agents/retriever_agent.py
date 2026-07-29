import os
from groq import Groq
from dotenv import load_dotenv

from rag.vector_store import query_vector_store


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def retrieve_and_explain(risk_flag):
    """Retrieve relevant XAI literature for the given risk flag and generate a draft explanation using Groq."""

    # 1. Build RAG query from risk factors
    query = (
        f"Explain financial risk factors: "
        f"{', '.join(risk_flag['factors'])}"
    )


    # 2. Retrieve relevant research papers
    context_results = query_vector_store(
        query,
        n_results=5
    )


    retrieved_chunks = context_results["documents"][0]

    context_text = "\n\n".join(
        retrieved_chunks
    )


    # 3. Build reasoning prompt
    prompt = f"""
You are an XAI explanation assistant.

A student has been flagged with:

Risk Level:
{risk_flag['risk_level']}

Contributing Factors:
{', '.join(risk_flag['factors'])}


Relevant research literature:

{context_text}


Based on the literature above, explain why this student may be at financial risk.

The explanation should:
- Be human understandable
- Use explainable AI concepts
- Mention important contributing factors
- Relate reasoning to feature importance,
  SHAP or LIME style explanations where relevant
"""


    # 4. Generate explanation using Groq LLM
    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.4
    )


    draft_explanation = (
        response
        .choices[0]
        .message
        .content
    )


    # 5. Agent-to-agent message format
    return {

        "student_id": risk_flag["student_id"],

        "risk_level": risk_flag["risk_level"],

        "retrieved_context": retrieved_chunks,

        "draft_explanation": draft_explanation
    }



# Testing Agent 1
if __name__ == "__main__":


    sample_risk_flag = {

        "student_id": "S1023",

        "risk_level": "High",

        "factors": [
            "low monthly income",
            "high dependency ratio",
            "delayed fee payments"
        ]
    }


    result = retrieve_and_explain(
        sample_risk_flag
    )


    print("\n--- Agent 1 Output ---\n")

    print(result["draft_explanation"])