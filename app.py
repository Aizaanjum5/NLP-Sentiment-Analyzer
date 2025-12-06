import streamlit as st
import joblib
import numpy as np
import re
import string
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== Download NLTK resources ==========
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords

# ========== Load Artifacts ==========
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("saved_models/tfidf_vectorizer.joblib")
    log_reg = joblib.load("saved_models/log_reg_model.joblib")
    bow_vect = joblib.load("saved_models/bow_vectorizer.joblib")
    nb = joblib.load("saved_models/nb_model.joblib")
    tokenizer = joblib.load("saved_models/tokenizer.joblib")
    label_encoder = joblib.load("saved_models/label_encoder.joblib")
    lstm_model = load_model("saved_models/lstm_model.h5")  # or .keras if you changed it
    return tfidf, log_reg, bow_vect, nb, tokenizer, label_encoder, lstm_model

tfidf, log_reg, bow_vect, nb, tokenizer, label_encoder, lstm_model = load_artifacts()

stop_words = set(stopwords.words("english"))

# ========== Text Cleaning ==========
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ========== Prediction Function ==========
def predict_sentiment(text):
    cleaned = clean_text(text)

    # Logistic Regression (TF-IDF)
    X_tfidf = tfidf.transform([cleaned])
    pred_lr = log_reg.predict(X_tfidf)[0]

    # Naive Bayes (BoW)
    X_bow = bow_vect.transform([cleaned])
    pred_nb = nb.predict(X_bow)[0]

    # LSTM
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
    proba = lstm_model.predict(pad)
    pred_lstm = label_encoder.inverse_transform([np.argmax(proba)])[0]

    return pred_lr, pred_nb, pred_lstm

# ========= Streamlit UI ==========
st.title("🎓 University Review Sentiment Analyzer")
st.write("**Compare Machine Learning vs. Deep Learning Models**")

st.markdown("""
This application analyzes university-related reviews and predicts whether
the text expresses a **positive**, **neutral**, or **negative** sentiment using:
- **Logistic Regression (TF-IDF)**
- **Naive Bayes (Bag-of-Words)**
- **LSTM (Word Embedding + Sequential Model)**
""")

user_input = st.text_area(" Enter a university/course review here:", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning(" Please enter some text to analyze.")
    else:
        pred_lr, pred_nb, pred_lstm = predict_sentiment(user_input)

        st.subheader(" Model Predictions")
        col1, col2, col3 = st.columns(3)
        col1.metric("Logistic Regression", pred_lr)
        col2.metric("Naive Bayes", pred_nb)
        col3.metric("LSTM", pred_lstm)

        st.write("---")
        st.subheader(" Model Insights")
        st.markdown("""
        ✔ **Logistic Regression** uses word frequency importance (**TF-IDF**).  
        ✔ **Naive Bayes** assumes word independence and is very fast.  
        ✔ **LSTM** reads sequences as ordered patterns, capturing context better.""")
