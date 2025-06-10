# src/preprocess.py
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, sep='\t', names=["label", "text"])
    df['text_clean'] = df['text'].apply(clean_text)
    return df

def vectorize(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text_clean'])
    y = df['label'].map({'ham': 0, 'spam': 1})
    return X, y, vectorizer

def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)