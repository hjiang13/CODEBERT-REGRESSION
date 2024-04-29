import os
from keybert import KeyBERT
from typing import List, Dict

def load_file_content(file_path: str) -> str:
    """Load and return the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_keywords(text: str, num_keywords: int = 20) -> List[str]:
    """Extract keywords from the provided text."""
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', use_mmr=True, diversity=0.5, top_n=num_keywords)
    return [kw[0] for kw in keywords]

def extract_features_from_files(directory: str) -> Dict[str, List[str]]:
    """Extract features from all .cpp files in the specified directory."""
    features = {}
    for filename in os.listdir(directory):
        if filename.endswith(".cpp"):
            file_path = os.path.join(directory, filename)
            content = load_file_content(file_path)
            keywords = extract_keywords(content)
            features[filename] = keywords
    return features

# Directory containing the C++ files
directory = '../DARE/hpc_applications/Benchmarks/'
features = extract_features_from_files(directory)

for file, keywords in features.items():
    print(f"File: {file}\nTop 20 Keywords: {keywords}\n")

import os
import csv
from keybert import KeyBERT
from typing import List, Dict

def load_file_content(file_path: str) -> str:
    """Load and return the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_keywords(text: str, num_keywords: int = 20) -> List[str]:
    """Extract keywords from the provided text."""
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', use_mmr=True, diversity=0.5, top_n=num_keywords)
    return [kw[0] for kw in keywords]

def extract_features_from_files(directory: str) -> Dict[str, List[str]]:
    """Extract features from all .cpp files in the specified directory and save to CSV."""
    features = {}
    for filename in os.listdir(directory):
        if filename.endswith(".cpp"):
            file_path = os.path.join(directory, filename)
            content = load_file_content(file_path)
            keywords = extract_keywords(content)
            base_filename = os.path.splitext(filename)[0]  # Remove the .cpp extension
            features[base_filename] = keywords
    return features

def save_features_to_csv(features: Dict[str, List[str]], csv_path: str):
    """Save the features dictionary to a CSV file."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Keywords'])
        for filename, keywords in features.items():
            writer.writerow([filename, ','.join(keywords)])

# Directory containing the C++ files
directory = '../DARE/hpc_applications/Benchmarks/'
# Path to save the CSV
csv_path = '../DARE/hpc_applications/Benchmarks/keyBERTFeatures.csv'

features = extract_features_from_files(directory)
save_features_to_csv(features, csv_path)

print(f"Data has been successfully saved to {csv_path}")