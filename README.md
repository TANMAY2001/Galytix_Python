# Project Documentation

## Folder Structure
```
Galytix_python/
│-- app.py
│-- GoogleNews-vectors-negative300.bin
│-- phrase_similarity.py
│-- phrases.csv
│-- Pipfile
│-- Pipfile.lock
│-- requirements.txt
│-- test_phrase_similarity.py
```

## Running Phrase Similarity Script
The `phrase_similarity.py` script runs examples and compares phrase similarities using a model and a phrase dataset.

### Usage:
```sh
python phrase_similarity.py --model GoogleNews-vectors-negative300.bin --phrases phrases.csv
```

To test with an additional input phrase:
```sh
python phrase_similarity.py --model GoogleNews-vectors-negative300.bin --phrases phrases.csv --input "insurance policy details"
```

## Running Tests
A separate test file `test_phrase_similarity.py` is provided. Run the tests using:
```sh
python -m pytest test_phrase_similarity.py
```

## Running FastAPI App
The FastAPI application is implemented in `app.py`. Start the FastAPI server using:
```sh
python -m uvicorn app:app --reload
```

## Dependencies
Ensure you have all required dependencies installed by running:
```sh
pip install -r requirements.txt
```

## Notes
- The `GoogleNews-vectors-negative300.bin` file is required for running phrase similarity calculations. Please download that from from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit and include that in the project folder.
- The FastAPI app provides an API for additional functionalities.

Feel free to modify and extend the functionality as needed! 🚀

