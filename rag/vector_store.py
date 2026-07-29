import os
import glob
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF file.
    """
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            full_text += page_text + "\n"

    return full_text


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    """
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk:
            chunks.append(chunk)

    return chunks


def build_vector_store():
    """
    Read PDF papers, create embeddings,
    and store them in ChromaDB.
    """

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="xai_papers"
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    pdf_files = glob.glob(
        "data/papers/*.pdf"
    )

    chunk_id = 0

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path}")

        text = extract_text_from_pdf(pdf_path)

        chunks = chunk_text(text)

        for chunk in chunks:

            embedding = model.encode(
                chunk
            ).tolist()

            collection.add(
                ids=[str(chunk_id)],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "source": os.path.basename(pdf_path)
                    }
                ]
            )

            chunk_id += 1


    print(
        f"Indexed {chunk_id} chunks from {len(pdf_files)} papers."
    )


def query_vector_store(query, n_results=5):
    """
    Search relevant chunks from vector database.
    """

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="xai_papers"
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    query_embedding = model.encode(
        query
    ).tolist()


    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results



# Temporary testing
if __name__ == "__main__":

    sample_pdf = "data/papers/YOUR_PDF_FILENAME.pdf"

    text = extract_text_from_pdf(sample_pdf)

    print(text[:500])