from datasets import load_dataset
import pickle
import os

os.makedirs('src/data', exist_ok=True)

datasets = {
    "arena": "lmsys/chatbot_arena_conversations",
    "dolly": "databricks/databricks-dolly-15k",
    "oasst1": "OpenAssistant/oasst1",
    "openorca": "Open-Orca/OpenOrca"
}

for name, path in datasets.items():
    print(f"Downloading {name} dataset...")
    dataset = load_dataset(path)
    with open(f'src/data/{name}_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved {name} dataset")

print("All datasets downloaded and saved.")