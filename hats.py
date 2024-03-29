import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import warnings
from collections import Counter

#streamlit run hate stream.py

warnings.filterwarnings('ignore')

# Data Loading (replace with your actual data loading logic)
data = pd.read_csv("train.csv")  # Assuming your CSV is named "train.csv"

# Data Preprocessing Functions
def remove_special_char(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)

def remove_urls(text):
    return re.sub(r"http\S+", "", text)

def remove_usernames_hashtags(text):
    return re.sub(r"@\w+|#\w+", "", text)

def remove_extra_spaces(text):
    return re.sub(r"\s+", " ", text)

def clean_text(text):
    text = text.lower()
    text = remove_special_char(text)
    text = remove_urls(text)
    text = remove_usernames_hashtags(text)
    text = remove_extra_spaces(text)
    return text

data["cleaned_tweet"] = data["tweet"].apply(clean_text)

# Feature Engineering (TF-IDF)
words = data["cleaned_tweet"].apply(lambda x: [word for word in x.split()])
word_count = Counter([word for sublist in words for word in sublist])

tfidf = TfidfVectorizer(max_features=5000)
X_train, X_test, y_train, y_test = train_test_split(
    data["cleaned_tweet"], data["class"], test_size=0.2, random_state=42
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Training
model_lr = LogisticRegression()
model_lr.fit(X_train_tfidf, y_train)

model_rf = RandomForestClassifier()
model_rf.fit(X_train_tfidf, y_train)

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_tfidf, y_train)

# User Input Section
st.title("Hate Speech Detection")
st.subheader("by Vishesh Joshi")
user_input = st.text_input("Enter Text to Analyse:")

if user_input:
    # Preprocess User Input
    cleaned_user_input = clean_text(user_input)
    user_input_tfidf = tfidf.transform([cleaned_user_input])
    
    # Model Predictions
    prediction_lr = model_lr.predict(user_input_tfidf)[0]
    prediction_rf = model_rf.predict(user_input_tfidf)[0]
    prediction_dt = model_dt.predict(user_input_tfidf)[0]
    if(prediction_lr==0):
        LogisticPreds = "Hate Speech Detected"
    elif(prediction_lr==1):
        LogisticPreds = "Offensive Language Detected"
    else:
        LogisticPreds = "Your text is Neutral"

    if(prediction_rf==0):
        RandomfPreds = "Hate Speech Detected"
    elif(prediction_rf==1):
        RandomfPreds = "Offensive Language Detected"
    else:
        RandomfPreds = "Your text is Neutral"

    if(prediction_dt==0):
        DecisionPreds = "Hate Speech Detected"
    elif(prediction_dt==1):
        DecisionPreds = "Offensive Language Detected"
    else:
        DecisionPreds = "Your text is Neutral"    
    
    

    # Display Predictions
    st.write("**Logistic Regression Prediction:**", LogisticPreds)
    st.write("**Random Forest Prediction:**", RandomfPreds)
    st.write("**Decision Tree Prediction:**", DecisionPreds)

    # Explanation (optional)
    class_labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
    explanation_text = f"**Explanation:**\nWhile the models provide predictions, it's important to consider the context and nuances of language. Use your judgment to evaluate the intent behind the text."
    st.write(explanation_text)
else:
    st.write("Please enter some text to analyze.")

# Streamlit Display Enhancements (optional)
st.sidebar.header("About the App")
st.sidebar.write(
    """This app uses machine learning models to classify text as hate speech, offensive language, or neutral. It's a work in progress, and the accuracy may vary. 
    Please use your judgment when interpreting the results."""
)
