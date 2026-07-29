import os
import glob
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "xai_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PAPERS_PATH = "data/papers"


# Load embedding model once
model = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """
    Extract raw text from every page of a PDF
    and return it as a single string.
    """

    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            full_text += page_text + "\n"

    return full_text.strip()


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping word-based chunks
    to preserve context between chunks.
    """

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk_words = words[
            start:start + chunk_size
        ]

        if not chunk_words:
            break

        chunks.append(
            " ".join(chunk_words)
        )

        if start + chunk_size >= len(words):
            break

    return chunks


# ---------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------

def build_vector_store():
    """
    Extract, chunk, embed and index all PDFs
    inside data/papers/ into ChromaDB.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    pdf_files = glob.glob(
        os.path.join(PAPERS_PATH, "*.pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return

    total_chunks = 0

    for pdf_path in pdf_files:

        filename = os.path.basename(pdf_path)

        print(f"Processing: {filename}")

        text = extract_text_from_pdf(pdf_path)

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            # Unique ID based on PDF + chunk number
            chunk_id = f"{filename}_{index}"

            # Avoid duplicate insertion
            existing = collection.get(
                ids=[chunk_id]
            )

            if existing["ids"]:
                continue

            embedding = model.encode(
                chunk
            ).tolist()

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "source": filename,
                        "chunk_index": index
                    }
                ]
            )

            total_chunks += 1

    print(
        f"Indexed {total_chunks} new chunks "
        f"from {len(pdf_files)} papers."
    )


# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

def query_vector_store(query, n_results=5):
    """
    Embed a query and return the top-N
    most relevant chunks from ChromaDB.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    query_embedding = model.encode(
        query
    ).tolist()

    n_results = min(
        n_results,
        collection.count()
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------

if __name__ == "__main__":

    build_vector_store()

    results = query_vector_store(
        "What is explainable artificial intelligence?",
        n_results=5
    )

    print("\nRetrieved Documents:\n")

    for i, document in enumerate(
        results["documents"][0]
    ):

        source = results["metadatas"][0][i]["source"]

        print(f"\n--- Result {i + 1} ---")
        print(f"Source: {source}")
        print(document[:500])