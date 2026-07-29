from rag.vector_store import query_vector_store

queries = [
    "How does SHAP explain credit risk predictions?",
    "What machine learning models are used for student dropout prediction?",
    "How does LIME provide local interpretability?",
    "What features are most important for predicting academic risk?",
    "How can explainable AI improve trust in financial risk models?"
]

for q in queries:
    print(f"\n{'='*60}\nQUERY: {q}\n{'='*60}")
    results = query_vector_store(q, n_results=3)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"--- from {meta['source']} ---")
        print(doc[:150])
        print()