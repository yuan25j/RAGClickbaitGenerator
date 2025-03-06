import csv
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def embedding_retrieve(query):
    model = SentenceTransformer("BAAI/bge-small-en")
    index = faiss.read_index("faiss_index.bin")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)

    query_embedding = model.encode([query], convert_to_numpy=True)
    N = 5  # Number of results
    distances, indices = index.search(query_embedding, N)
    print("\nTop similar headlines:")
    titles = [texts[indices[0][i]] for i in range(N)]
    return titles

def word_overlap_retrieve(paper_title: str, csv_file: str = 'clickbait_data.csv'):
    """
    Reads the CSV file with two columns: title and a label indicating if it is clickbait.
    Only the titles marked as clickbait are used.
    
    The CSV file may have a header row; this function attempts to detect and skip it.
    It then computes a simple matching score (based on word overlap) between the
    input paper title and each clickbait example, returning the top three examples.
    """
    examples = []
    valid_labels = {"clickbait", "yes", "true", "1"}
    
    try:
        with open(csv_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader, None)
            # Check if the first row is a header by testing if the second column is not a valid label.
            if first_row and first_row[1].strip().lower() not in valid_labels:
                # Header detected; continue with the remaining rows.
                pass
            else:
                # First row is data; process it.
                if first_row and first_row[1].strip().lower() in valid_labels:
                    examples.append(first_row[0].strip())
            
            for row in reader:
                if len(row) < 2:
                    continue
                label = row[1].strip().lower()
                if label in valid_labels:
                    examples.append(row[0].strip())
    except FileNotFoundError:
        print(f"CSV file '{csv_file}' not found.")
        return []
    
    # Compute a simple similarity score (word overlap) between the input title and each example.
    paper_words = set(paper_title.lower().split())
    scored_examples = []
    for ex in examples:
        ex_words = set(ex.lower().split())
        score = len(paper_words.intersection(ex_words))
        scored_examples.append((score, ex))
    
    # Sort examples by score (highest first) and return the top three.
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    top_examples = [ex for score, ex in scored_examples[:3]]
    
    # Fallback: if no examples match by keywords, return up to three examples from the file.
    if not top_examples and examples:
        top_examples = examples[:3]
    
    return top_examples
