import faiss
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def test_retrieval(query_text):
    print("Loading embedding model...")
    # We must use the exact same model we used for indexing
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    # Disable the default LLM so we don't get an OpenAI API key error
    Settings.llm = None

    print("Loading FAISS index from disk...")
    # Point LlamaIndex to your saved FAISS folder
    vector_store = FaissVectorStore.from_persist_dir("./legal_faiss_index")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir="./legal_faiss_index"
    )

    # Correctly rebuild the index from the saved docstore and vector store
    index = load_index_from_storage(storage_context)

    print(f"\nSearching for: '{query_text}'...")
    # Configure the retriever to bring back the top 2 matches
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve(query_text)

    print("\n--- Top Matches Found ---")
    for i, node in enumerate(nodes):
        print(f"\nMatch {i+1} (Relevance Score: {node.score:.4f}):")
        # Print the first 500 characters of the matched document
        print(node.text[:500] + "...\n" + "-"*40)


if __name__ == "__main__":
    # Feel free to change this search term to anything!
    test_retrieval("stolen property and chop shop")
