# Fake News Detection — TF-IDF + Machine Learning + Streamlit

An end-to-end Python project for classifying news text as **REAL** or **FAKE/UNRELIABLE** using classical machine learning.

## Features

- Text cleaning
- Stopword removal
- Selectable stemming or lemmatization
- TF-IDF feature extraction
- Multiple models:
  - Logistic Regression
  - Linear SVM
  - Multinomial Naive Bayes
  - Passive Aggressive Classifier
- Hyperparameter tuning using 5-fold GridSearchCV
- Evaluation with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Classification report
  - Confusion matrix
- Saved model with Joblib
- Streamlit web interface

## Recommended CSV format

```csv
title,text,label
"Example title","Full article text here...",1
"Another title","Another article...",0
```

By default the training script assumes:

- `1 = FAKE`
- `0 = REAL`

If your dataset uses labels `FAKE` and `REAL`, it is detected automatically.

If your numeric dataset uses a different value for fake news, use `--fake-label`.

## Project structure

```text
fake_news_detection/
├── app.py
├── train.py
├── utils.py
├── requirements.txt
├── README.md
├── data/
│   └── news.csv
└── artifacts/
    ├── fake_news_model.joblib       # created after training
    ├── metrics.json                 # created after training
    ├── model_comparison.csv         # created after training
    └── confusion_matrix.png         # created after training
```

## 1. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Put your dataset in `data/news.csv`

The CSV needs a `label` column and at least one of:

- `text`
- `title`

If both are present, they are combined.

## 4. Train with lemmatization

```bash
python train.py --data data/news.csv --preprocess lemmatization --fake-label 1
```

Or compare by training again with stemming:

```bash
python train.py --data data/news.csv --preprocess stemming --fake-label 1
```

The run selects the model by cross-validation F1-score, then writes the model and evaluation files to `artifacts/`.

## 5. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Important project note

This model learns statistical language patterns from the training dataset. A high score does **not** prove that a real-world claim is factually true or false. For a stronger future version, add retrieval from trusted news/fact-check sources and claim-evidence verification.
