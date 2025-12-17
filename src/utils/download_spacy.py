import spacy
from spacy.cli import download
import os

save_path = "models/spacy_en_model"
if not os.path.exists(save_path):
    os.makedirs(save_path)

download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

nlp.to_disk(save_path)

print(f"Success! Model saved to: {save_path}")