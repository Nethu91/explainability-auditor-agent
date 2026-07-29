from rag.vector_store import query_vector_store

results = query_vector_store("How does SHAP explain credit risk predictions?")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"--- from {meta['source']} ---")
    print(doc[:200])
    print()