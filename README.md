# Explainability Auditor

An agentic AI system that explains student financial-risk predictions by grounding its reasoning in explainable-ML (XAI) literature. Built for IT41043 — Intelligent Systems (Agentic AI), Horizon Campus.

**Live demo:** https://explainability-auditor-agent-ve7cpuqca2os9kf39jyz3z.streamlit.app/

## Project Description

This tool takes a student's financial-risk flag (risk level + contributing factors) and produces a human-readable explanation of *why* the student was flagged — grounded in real XAI research (SHAP, LIME, feature-importance literature) rather than a generic LLM guess. A second agent then critiques the explanation against the retrieved literature, catching unsupported claims before the final output is shown.

This was built to support ongoing research on explainable ML for predicting financial risk among Sri Lankan university students — the tool helps generate and sanity-check literature-grounded explanations for model outputs.

## Architecture

[User Input: risk_flag]
│
▼
[Agent 1: Retriever/Explainer] ──uses──> [RAG: ChromaDB + sentence-transformers]
│ (Groq llama-3.1-8b-instant) (10 XAI/financial-risk papers, 136 chunks)
│
│ draft_explanation + retrieved_context
▼
[Agent 2: Critic/Validator] (Groq llama-3.3-70b-versatile)
│
│ {approved, issues, revised_explanation}
▼
[Streamlit UI: displays draft, critique, and final explanation]

**Agentic design patterns used:**
1. **Tool-use** — Agent 1 calls the RAG retrieval tool (`query_vector_store`) to fetch grounding literature before generating an explanation.
2. **ReAct-style reasoning** — Agent 1 combines retrieved context with the risk flag to reason about *why* the student is at risk before producing output (see `agents/retriever_agent.py`, `retrieve_and_explain()`).
3. **Reflection / self-critique** — Agent 2 reviews Agent 1's draft against the same retrieved literature, flags unsupported claims, and produces a revised explanation (see `agents/critic_agent.py`, `critique_explanation()`).

## Agent-to-Agent Communication

Agent 1 and Agent 2 communicate via a structured JSON message (no external framework — a custom protocol):

Agent 1 output → Agent 2 input:
{
"student_id": str,
"risk_level": str,
"retrieved_context": [str, ...],
"draft_explanation": str
}

Agent 2 output:
{
"student_id": str,
"approved": bool,
"issues": str,
"revised_explanation": str
}

**Sequence diagram:**
User → Streamlit → Agent1.retrieve_and_explain(risk_flag)
→ RAG.query_vector_store(query)
← retrieved_chunks
→ Groq(8B).generate(prompt)
← draft_explanation
Streamlit ← agent1_result
Streamlit → Agent2.critique_explanation(agent1_result)
→ Groq(70B).generate(critique_prompt)
← approved / issues / revised_explanation
Streamlit ← agent2_result → displayed to user

## Model Selection Strategy

| Sub-task | Model (provider) | Why chosen |
|---|---|---|
| Retrieval + draft explanation generation | Llama 3.1 8B Instant (Groq) | Very low latency, cheap, sufficient reasoning quality for drafting an explanation once relevant context has already been retrieved |
| Critique / validation of the draft | Llama 3.3 70B Versatile (Groq) | Larger model with stronger reasoning needed to judge whether claims are actually supported by the literature — catching hallucinations requires more careful reasoning than generating the first draft |

*(Both models are served via Groq — the deliberate choice was between model **size/capability**, not provider, since Groq's low latency made it well suited for both a fast drafting step and a heavier critique step, keeping the whole pipeline fast enough for an interactive Streamlit demo.)*

## RAG Pipeline

- **Corpus:** 10 papers on explainable ML (SHAP, LIME) and financial/academic risk prediction, sourced from arXiv (e.g. *Explainable AI in Credit Risk Management*, *Who will dropout from university? Academic risk prediction based on interpretable machine learning*, *Evaluating the explainers: black-box ML for student success prediction*).
- **Chunking:** PDF text extracted with `pypdf`, split into ~500-word chunks with 50-word overlap to preserve context across chunk boundaries.
- **Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers) — lightweight, fast, free.
- **Vector store:** ChromaDB (persistent local store), 136 chunks indexed total.

**Retrieval evaluation (5 sample queries):**

| Query | Retrieved relevant? |
|---|---|
| "How does SHAP explain credit risk predictions?" | ✅ Yes — returned SHAP/credit-risk chunks from the credit-risk XAI paper |
| "What ML models are used for student dropout prediction?" | ✅ Yes — returned dropout-prediction methodology chunks |
| "How does LIME provide local interpretability?" | ✅ Yes — returned LIME-vs-SHAP comparison chunks |
| "What features are most important for predicting academic risk?" | ✅ Yes — returned SHAP-value/feature-importance chunks from the academic risk paper |
| "How can explainable AI improve trust in financial risk models?" | ✅ Yes — returned XAI-in-credit-risk-management chunks discussing trust and regulation |

All 5 queries returned literature genuinely relevant to the question, confirming the retrieval pipeline is working correctly.

## Setup Instructions

```bash
git clone https://github.com/Nethu91/explainability-auditor-agent.git
cd explainability-auditor-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with:
GROQ_API_KEY=gsk_dkRXI95PyMYisAMMn...................................
GROQ_API_KEY=gsk_2nJ6YsKRxaXQM0ma....................................

Build the vector store, then run the app:
```bash
python test_build.py
streamlit run app.py
```

## Known Limitations

- The underlying financial-risk ML model (student risk classifier) is not yet trained; this system currently takes **simulated/mocked risk-flag inputs** rather than live model predictions. Integration with the actual trained model is planned future work.
- The RAG corpus (10 papers) is smaller than the suggested 20+; papers were curated for strong topical relevance (XAI methods + financial/academic risk prediction) rather than padded with lower-relevance sources.
- Agent 2's critique parsing relies on the LLM following a fixed output format (`APPROVED:` / `ISSUES:` / `REVISED_EXPLANATION:`); occasional format drift could break parsing — this could be made more robust with structured JSON output in future iterations.
- Both agents currently use Groq; OpenRouter integration was scoped out due to time constraints, though the architecture supports adding it as a third model source.

