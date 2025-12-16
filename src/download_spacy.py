import spacy
from spacy.cli import download
import os

# Ensure the directory exists
save_path = "models/spacy_en_model"
if not os.path.exists(save_path):
    os.makedirs(save_path)

print("Downloading model...")
# Download the model explicitly
download("en_core_web_sm")

print("Loading and saving to disk...")
# Load the system model
nlp = spacy.load("en_core_web_sm")

# Save it to your local folder
nlp.to_disk(save_path)

print(f"Success! Model saved to: {save_path}")