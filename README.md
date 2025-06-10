
# SMS Spam Filter

This project is a binary text classification system that detects spam messages from short SMS texts using machine learning. It is built with `scikit-learn`, `TF-IDF` vectorization, and deployed via a user-friendly `Streamlit` web interface.

---

## Features

- Classifies messages as **Spam** or **Ham** (not spam)
- Adjustable **Spam Risk Level** to change filtering strictness
- **Confidence score** shown for every prediction
- Fast and interpretable model using **Logistic Regression**
- Full analysis and visualizations in a **Jupyter Notebook**
- **Streamlit GUI** for real-time use by non-technical users

---

## Folder Structure

```
spam_filter_project/
│
├── data/
│   └── SMSSpamCollection.csv              # Dataset
│
├── models/
│   └── spam_filter_model.pkl              # Saved model + vectorizer
│
├── notebooks/
│   └── spam_filter_analysis.ipynb         # Full exploratory and modeling notebook
│
├── src/
│   ├── preprocess.py                      # Text cleaning & vectorization
│   ├── train_model.py                     # Model training & evaluation
│   └── risk_filter.py                     # Load model & threshold-based prediction
│
├── app/
│   └── gui_app.py                         # Streamlit GUI app
│
├── gui/
│   └── interface_proposal.md              # GUI integration proposal
│
├── reports/
│   └── charts/                            # Wordclouds, evaluation images (optional)
│
├── requirements.txt                       # Dependencies
└── README.md                              # This file
```

---

## Installation

1. Clone the repository or unzip the project folder.
2. Navigate to the project directory.
3. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

4. Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Train the Model

Train the model and save it to `models/spam_filter_model.pkl`:

```bash
python src/train_model.py
```

You will see:
- Classification report (precision, recall, F1)
- Confusion matrix
- Confirmation that the model was saved

---

## How to Launch the GUI

Launch the real-time Streamlit app:

```bash
streamlit run app/gui_app.py
```

Then open the provided localhost URL in your browser to test the filter.

---

## Requirements

```
pandas
scikit-learn
matplotlib
seaborn
wordcloud
nltk
streamlit
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Example Usage

**Message:** "Congratulations! You've won a free iPhone. Click now to claim."

- Low Risk (0.9) → Likely HAM
- Medium Risk (0.7) → HAM / borderline
- High Risk (0.5) → SPAM (Confidence: 0.92)

---

## Acknowledgements

- Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- GUI: [Streamlit](https://streamlit.io/)

---

## License

This project was developed as part of the **DLBDSME01 Model Engineering** case study submission for IU University.
