import faiss
import mlflow
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ingest_cap import fetch_cold_cases

mlflow.set_experiment("LegisAI_FAISS_Indexing")


def build_local_index():
    with mlflow.start_run(run_name="Build_FAISS_Index"):
        print("Fetching clean legal documents...")
        # Fetch 10 cases to start building our database
        docs = fetch_cold_cases(10)

        print("Initializing local embedding model...")
        # Use a high-quality, open-source embedding model
        # BGE-small has an output dimension size of 384
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = embed_model
        d = 384

        print("Setting up FAISS Vector Store...")
        # Initialize a flat L2 distance FAISS index
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)

        print("Converting text to vectors and indexing... (This may take a moment)")
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            show_progress=True
        )

        print("Saving FAISS index to disk...")
        index.storage_context.persist(persist_dir="./legal_faiss_index")
        print("Success! Your local vector database is built and saved.")


if __name__ == "__main__":
    build_local_index()
