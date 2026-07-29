from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


def critique_explanation(agent1_output):

    context_text = "\n\n".join(
        agent1_output["retrieved_context"]
    )


    prompt = f"""
You are a critical reviewer checking an AI-generated explanation for accuracy.

Draft explanation:
{agent1_output['draft_explanation']}


Source literature it should be grounded in:

{context_text}


Check whether every claim in the draft explanation is actually supported
by the source literature above.

Respond in this exact format:

APPROVED: yes/no

ISSUES:
<list any claims not supported by the literature, or "none">

REVISED_EXPLANATION:
<a corrected version that removes unsupported claims>
"""


    response = client.chat.completions.create(

        model="anthropic/claude-3.5-sonnet",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )


    output_text = (
        response
        .choices[0]
        .message
        .content
    )


    # Parse response
    approved_line = (
        output_text
        .split("APPROVED:")[1]
        .split("\n")[0]
        .strip()
        .lower()
    )


    approved = approved_line == "yes"


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


    return {

        "student_id": agent1_output["student_id"],

        "approved": approved,

        "issues": issues,

        "revised_explanation": revised
    }



# Testing Agent 2
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


    # Agent 1 output
    agent1_result = retrieve_and_explain(
        sample_risk_flag
    )


    # Agent 2 validation
    agent2_result = critique_explanation(
        agent1_result
    )


    print("\n--- Agent 2 (Critic) Output ---\n")


    print(
        "Approved:",
        agent2_result["approved"]
    )


    print(
        "\nIssues:",
        agent2_result["issues"]
    )


    print(
        "\nRevised:",
        agent2_result["revised_explanation"]
    )