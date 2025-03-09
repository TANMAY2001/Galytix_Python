import pytest
import numpy as np
from gensim.models import KeyedVectors
from phrase_similarity import get_phrase_vector, find_closest_match, compute_phrase_similarities_from_file

# Mock Word2Vec Model for Testing
class MockWord2Vec:
    def __init__(self):
        self.key_to_index = {"hello": 0, "world": 1}
        self.vectors = {
            "hello": np.array([0.1, 0.2, 0.3]),
            "world": np.array([0.4, 0.5, 0.6]),
        }
    
    def __contains__(self, key):
        return key in self.vectors
    
    def __getitem__(self, key):
        return self.vectors[key]

mock_model = MockWord2Vec()

# Test 1: Verify phrase vector computation (average of word vectors)
def test_get_phrase_vector():
    phrase_vector = get_phrase_vector("hello world", mock_model)
    expected_vector = np.array([0.25, 0.35, 0.45])  # Mean of "hello" & "world"
    assert phrase_vector is not None, "Phrase vector should not be None"
    assert np.allclose(phrase_vector, expected_vector), "Phrase vector computation is incorrect"

# Test 2: Ensure get_phrase_vector returns None for unknown words
def test_get_phrase_vector_unknown_word():
    phrase_vector = get_phrase_vector("unknownword", mock_model)
    assert phrase_vector is None, "Phrase vector should be None for unknown words"

# Test 3: Check if find_closest_match returns the most similar phrase
def test_find_closest_match():
    test_phrases = ["hello world", "goodbye moon"]
    with open("test_vectors.csv", "w") as f:
        f.write("hello world,0.25,0.35,0.45\n")
        f.write("goodbye moon,0.5,0.5,0.5\n")
    
    match, similarity = find_closest_match("hello", "test_vectors.csv", mock_model)
    assert match == "hello world", "Closest match function failed"
    assert similarity > 0, "Similarity should be positive"

# Test 4: Ensure find_closest_match returns None for empty input
def test_find_closest_match_empty():
    match, similarity = find_closest_match("", "test_vectors.csv", mock_model)
    assert match is None, "Match should be None for empty input"
    assert similarity is None, "Similarity should be None for empty input"

# Test 5: Check batch similarity computation
def test_compute_phrase_similarities():
    with open("test_vectors.csv", "w") as f:
        f.write("hello world,0.25,0.35,0.45\n")
        f.write("goodbye moon,0.5,0.5,0.5\n")
    
    results = compute_phrase_similarities_from_file("test_vectors.csv")
    assert len(results) > 0, "Similarity results should not be empty"
    assert results[0][2] >= 0, "Similarity score should be non-negative"
