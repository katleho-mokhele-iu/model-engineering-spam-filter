import pickle

def load_model(path="models/spam_filter_model.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)

def classify_message(text, model, vectorizer, threshold=0.5):
    import re, string
    text_clean = re.sub(r"http\S+|\d+", "", text.lower()).translate(str.maketrans('', '', string.punctuation))
    X = vectorizer.transform([text_clean])
    proba = model.predict_proba(X)[0][1]
    return ("spam", proba) if proba >= threshold else ("ham", proba)