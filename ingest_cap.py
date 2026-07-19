import mlflow
from datasets import load_dataset
from llama_index.core import Document

mlflow.set_experiment("LegisAI_Data_Ingestion")


def fetch_cold_cases(num_records=5):
    with mlflow.start_run(run_name="COLD_Cases_Stream", nested=True):
        print("Connecting to Hugging Face to fetch COLD Cases data...")

        # Load the ungated COLD Cases dataset
        dataset = load_dataset("harvard-lil/cold-cases",
                               split="train", streaming=True)

        documents = []
        for i, row in enumerate(dataset):
            if i >= num_records:
                break

            # 1. Extract and combine all opinion texts for the case
            opinions_list = row.get('opinions', [])
            full_opinion_text = ""

            for opinion in opinions_list:
                # Extract just the raw text string from the dictionary
                op_text = opinion.get('opinion_text', '')
                if op_text:
                    full_opinion_text += op_text + "\n\n"

            # 2. Build the clean document structure
            text_content = f"Case Name: {row.get('case_name', 'Unknown')}\n" \
                           f"Court: {row.get('court_full_name', 'Unknown')}\n" \
                           f"Date Filed: {row.get('date_filed', 'Unknown')}\n\n" \
                           f"Opinion:\n{full_opinion_text.strip()}"

            documents.append(Document(text=text_content))

        mlflow.log_metric("successfully_loaded_docs", len(documents))
        print(
            f"Successfully loaded {len(documents)} clean COLD case documents.")
        return documents


if __name__ == "__main__":
    docs = fetch_cold_cases(1)
    print("\n--- First Parsed Clean Document Snippet ---")
    print(docs[0].text[:800] + "...\n")
