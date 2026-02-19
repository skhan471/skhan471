## NLP Text Classification Pipeline (SMS Spam)

This repository now includes a complete, production-ready Python pipeline for binary text classification using the **SMS Spam Collection** dataset.

### Files
- `nlp_text_classification_pipeline.py`: End-to-end executable pipeline.
- `requirements.txt`: Python dependencies.

### What the script does
1. Downloads the public SMS Spam dataset from UCI.
2. Preprocesses text:
   - lowercasing
   - punctuation removal
   - stopword removal
   - tokenization
   - optional lemmatization (enabled)
3. Builds **TF-IDF** features.
4. Saves numerical TF-IDF data to Excel and reloads it.
5. Splits data into train/validation/test as 70%/15%/15%.
6. Trains Logistic Regression and evaluates on validation + test sets.
7. Builds **Word2Vec average sentence embeddings**.
8. Saves Word2Vec features to Excel and reloads.
9. Trains and evaluates a second Logistic Regression model.
10. Prints a comparison table for TF-IDF vs Word2Vec.

### Setup
```bash
python -m pip install -r requirements.txt
```

### Run
```bash
python nlp_text_classification_pipeline.py
```

### Expected output format
The script prints:
- dataset info and sample cleaned text,
- split sizes,
- metrics for each model:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- final comparison table:

```text
===== Comparison Table (TF-IDF vs Word2Vec) =====
                              Model  Val Accuracy  Test Accuracy  Precision  Recall     F1
      Logistic Regression + TF-IDF        0.98xx         0.97xx     0.9xxx  0.8xxx  0.8xxx
Logistic Regression + Word2Vec(avg)       0.9xxx         0.9xxx     0.8xxx  0.7xxx  0.7xxx
```

Generated Excel artifacts are saved under `artifacts/`.
