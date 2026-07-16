import mlflow
from datasets import load_dataset
from llama_index.core import Document

# Initialize MLflow Experiment
mlflow.set_experiment("LegisAI_Data_Ingestion")


def fetch_contract_data(num_records=5):
    # Start an MLflow tracking run
    with mlflow.start_run(run_name="CUAD_Data_Fetch"):
        print("Connecting to Hugging Face to fetch open CUAD data...")

        # Log parameters to MLflow
        mlflow.log_param("num_records_requested", num_records)
        mlflow.log_param("dataset_source", "theatticusproject/cuad")

        # Load the open dataset in streaming mode
        dataset = load_dataset("theatticusproject/cuad",
                               split="train", streaming=True)

        documents = []
        for i, row in enumerate(dataset):
            if i >= num_records:
                break

            print(f"Extracting text from PDF {i+1}...")

            # 1. Grab the raw PDF object
            pdf_obj = row['pdf']

            # 2. Iterate through its pages to extract the text
            full_text = ""
            for page in pdf_obj.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

            # 3. Structure it into a LlamaIndex Document
            title = f"CUAD_Contract_{i+1}"
            text_content = f"Contract Title: {title}\n\nContract Text:\n{full_text}"

            documents.append(Document(text=text_content))

        # Log metrics to MLflow
        mlflow.log_metric("successfully_loaded_docs", len(documents))

        print(f"\nSuccessfully loaded {len(documents)} contract documents.")
        return documents


if __name__ == "__main__":
    # Fetch just 1 contract to test the extraction
    docs = fetch_contract_data(1)
    print("\n--- First Parsed Contract Snippet ---")
    # Print the first 800 characters to verify successful extraction
    print(docs[0].text[:800] + "...\n")
