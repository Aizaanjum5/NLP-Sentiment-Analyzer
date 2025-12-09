 University Review Sentiment Analyzer
 Project Overview

This project analyzes university or course reviews and predicts whether the sentiment is:
-Positive
-Neutral
-Negative

The goal is to compare **classical machine learning models** with a **deep learning model** on a text classification task.

---
 Models Used

1. **Logistic Regression** (with TF-IDF features)  
2. **Multinomial Naive Bayes** (with Bag-of-Words features)  
3. **LSTM Neural Network** (with word embeddings and sequence modeling)

---

NLP Pipeline

- Text cleaning (lowercasing, punctuation removal, stopword removal)
- Tokenization
- Train / validation / test split
- Vectorization (TF-IDF, Bag-of-Words, sequences for LSTM)
- Model training and evaluation


Dataset

- Synthetic dataset of ~4,500 reviews
- 3 sentiment labels: `positive`, `neutral`, `negative`
- Reviews talk about:
  - professors
  - lectures
  - assignments
  - exams
  - projects
  - workload

The dataset was generated programmatically to simulate realistic student feedback 
Application (Streamlit)

The app allows the user to:

1. Type a university or course review
2. See predictions from all three models:
   - Logistic Regression
   - Naive Bayes
   - LSTM
3. Compare how classical ML and deep learning behave on the same input.

Run the app with:

```bash
pip install -r requirements.txt
streamlit run app.py
