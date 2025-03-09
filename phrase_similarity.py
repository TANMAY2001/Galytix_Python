import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import argparse
import csv
import Levenshtein

# Download stopwords if needed
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load the pre-trained Word2Vec model (first million words)
def load_word2vec_model(model_path: str):
    print("Loading Word2Vec model...")
    wv = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=1000000)
    # Save first million words to vectors.csv
    print("Saving word vectors to vectors.csv...")
    wv.save_word2vec_format("vectors.csv")
    return wv

# Clean stopwords and duplicates from a phrase
def clean_phrase(phrase):
    words = phrase.lower().split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = list(set(words))  # Remove duplicates
    return " ".join(words)

# Handle OOV words using Levenshtein distance
def get_closest_word(word, model):
    if word in model:
        return word
    closest_word = min(model.key_to_index.keys(), key=lambda w: Levenshtein.distance(word, w))
    return closest_word if Levenshtein.distance(word, closest_word) <= 2 else None

# Compute phrase vector with cleaned words & OOV handling
def get_phrase_vector(phrase: str, model: KeyedVectors):
    cleaned_phrase = clean_phrase(phrase)
    words = cleaned_phrase.split()

    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
        else:
            closest_word = get_closest_word(word, model)
            if closest_word:
                vectors.append(model[closest_word])  # Use closest word's vector
    
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

# Save phrase vectors to a file instead of keeping them in memory
def save_phrase_vectors(phrases, model, output_file="phrase_vectors.csv"):
    print("Saving phrase vectors to file...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for phrase in phrases:
            vector = get_phrase_vector(phrase, model)
            if vector is not None:
                writer.writerow([phrase] + vector.tolist())

# Compute similarity from saved phrase vectors
def compute_phrase_similarities_from_file(phrase_vectors_file):
    print("Computing similarities from saved vectors...")
    phrases = []
    vectors = []
    
    # Read vectors from file
    with open(phrase_vectors_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            phrases.append(row[0])
            vectors.append(np.array(row[1:], dtype=np.float32))

    # Convert list of vectors into a NumPy matrix
    matrix = np.array(vectors)  # Shape: (num_phrases, 300)
    
    # Compute similarity using matrix multiplication
    similarity_matrix = matrix @ matrix.T  # Shape: (num_phrases, num_phrases)

    results = []
    num_phrases = len(phrases)
    for i in range(num_phrases):
        for j in range(i + 1, num_phrases):  # Avoid duplicate comparisons
            results.append((phrases[i], phrases[j], similarity_matrix[i, j]))

    return results

# Find the closest phrase using precomputed vectors
def find_closest_match(input_phrase: str, phrase_vectors_file, model):
    input_vector = get_phrase_vector(input_phrase, model)
    if input_vector is None:
        return None, None
    
    closest_match = None
    best_similarity = -1

    with open(phrase_vectors_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            phrase = row[0]
            vector = np.array(row[1:], dtype=np.float32)
            similarity = 1 - cosine(input_vector, vector)
            if similarity > best_similarity:
                best_similarity = similarity
                closest_match = phrase

    return closest_match, best_similarity

# Command line interface
def main():
    parser = argparse.ArgumentParser(description="Phrase similarity calculator")
    parser.add_argument("--model", type=str, required=True, help="Path to Word2Vec model")
    parser.add_argument("--phrases", type=str, required=True, help="Path to phrases CSV file")
    parser.add_argument("--input", type=str, help="Input phrase for similarity search")
    args = parser.parse_args()
    
    model = load_word2vec_model(args.model)
    
    phrases_df = pd.read_csv(args.phrases, encoding="ISO-8859-1")
    phrases = phrases_df.iloc[:, 0].tolist()
    
    # Save phrase vectors to file instead of keeping in memory
    save_phrase_vectors(phrases, model)
    
    if args.input:
        match, similarity = find_closest_match(args.input, "phrase_vectors.csv", model)
        print(f"Closest match: {match} with similarity: {similarity}")
    else:
        similarities = compute_phrase_similarities_from_file("phrase_vectors.csv")
        print("Computed phrase similarities:", similarities)

if __name__ == "__main__":
    main()