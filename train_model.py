import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------- 1. Load the Dataset ----------
print("Loading dataset...")
df = pd.read_csv("UpdatedResumeDataSet.csv")  # put dataset in same folder

# ---------- 2. Clean Text Function ----------
def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(tokens)

print("Cleaning text data...")
df['Resume'] = df['Resume'].apply(clean_text)

# ---------- 3. Encode Target Labels ----------
print("Encoding categories...")
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# ---------- 4. TF-IDF Vectorization ----------
print("Vectorizing text...")
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Resume'])
y = df['Category']

# ---------- 5. Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------- 6. Train Model ----------
print("Training model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ---------- 7. Evaluate ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# ---------- 8. Save the Model and Transformers ----------
print("Saving model and preprocessing objects...")
joblib.dump(model, "resume_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Files saved: resume_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl")
