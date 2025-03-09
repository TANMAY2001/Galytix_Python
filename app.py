from fastapi import FastAPI
import csv
from phrase_similarity import (
    load_word2vec_model,
    save_phrase_vectors,
    find_closest_match
)

# File paths
MODEL_PATH = "GoogleNews-vectors-negative300.bin"
PHRASE_FILE = "phrases.csv"
VECTOR_FILE = "phrase_vectors.csv"

# Initialize FastAPI app
app = FastAPI()

# Global model variable
model = None

@app.on_event("startup")
def startup_event():
    """Load the model and compute phrase vectors on app startup."""
    global model
    model = load_word2vec_model(MODEL_PATH)

    # Load phrases from CSV
    with open(PHRASE_FILE, "r", encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        phrases = [row[0] for row in reader]

    # Compute and save phrase vectors
    save_phrase_vectors(phrases, model, VECTOR_FILE)
    print("Phrase vectors computed and saved!")


@app.get("/")
def home():
    return {"message": "Phrase Similarity API is running"}


@app.get("/similarity/")
def find_similar(input_phrase: str):
    """Find the most similar phrase."""
    match, similarity = find_closest_match(input_phrase, VECTOR_FILE, model)
    if match:
        return {"input": input_phrase, "closest_match": match, "similarity": round(similarity, 4)}
    return {"input": input_phrase, "message": "No similar phrase found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
