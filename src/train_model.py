# src/train_model.py
import pickle
from preprocess import load_and_preprocess, vectorize, train_test_split_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def train(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def save_model(model, vectorizer, path="models/spam_filter_model.pkl"):
    with open(path, 'wb') as f:
        pickle.dump((model, vectorizer), f)

if __name__ == "__main__":
    df = load_and_preprocess("data/SMSSpamCollection.csv")
    X, y, vectorizer = vectorize(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model, vectorizer)
    print("Model training complete and saved to 'models/spam_filter_model.pk1'")