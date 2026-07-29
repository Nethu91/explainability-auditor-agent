import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def critique_explanation(agent1_output):
    """
    Agent 2:
    Reviews Agent 1's explanation against the retrieved literature
    and produces a validated explanation.
    """

    context_text = "\n\n".join(
        agent1_output["retrieved_context"]
    )

    prompt = f"""
You are an expert AI explanation validator.

Your task is to critically review an AI-generated explanation.

Student ID:
{agent1_output["student_id"]}

Draft Explanation:
{agent1_output["draft_explanation"]}

Retrieved Literature:
{context_text}

Instructions:

1. Check whether every important claim is supported by the literature.
2. Identify unsupported or exaggerated statements.
3. If necessary, rewrite the explanation so that it is completely grounded in the literature.
4. Keep the explanation easy for non-technical users to understand.

Respond EXACTLY in this format:

APPROVED: yes/no

ISSUES:
- issue 1
- issue 2

REVISED_EXPLANATION:
<your corrected explanation>
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    output_text = response.choices[0].message.content

    # Parse model output safely
    try:

        approved = (
            output_text
            .split("APPROVED:")[1]
            .split("\n")[0]
            .strip()
            .lower() == "yes"
        )

        issues = (
            output_text
            .split("ISSUES:")[1]
            .split("REVISED_EXPLANATION:")[0]
            .strip()
        )

        revised = (
            output_text
            .split("REVISED_EXPLANATION:")[1]
            .strip()
        )

    except Exception:

        approved = False
        issues = "Unable to parse model response."

        revised = output_text

    return {

        "student_id": agent1_output["student_id"],

        "approved": approved,

        "issues": issues,

        "revised_explanation": revised
    }


if __name__ == "__main__":

    from agents.retriever_agent import retrieve_and_explain

    sample_risk_flag = {

        "student_id": "S1023",

        "risk_level": "High",

        "factors": [
            "low monthly income",
            "high dependency ratio",
            "delayed fee payments"
        ]
    }

    print("\nRunning Agent 1...\n")

    agent1_result = retrieve_and_explain(
        sample_risk_flag
    )

    print("Running Agent 2...\n")

    agent2_result = critique_explanation(
        agent1_result
    )

    print("\n========== AGENT 2 OUTPUT ==========\n")

    print("Student ID :", agent2_result["student_id"])
    print("Approved  :", agent2_result["approved"])

    print("\nIssues:")
    print(agent2_result["issues"])

    print("\nRevised Explanation:\n")
    print(agent2_result["revised_explanation"])