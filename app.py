from flask import Flask, render_template, request, redirect, url_for
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os
import PyPDF2

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    resume_text = request.form.get('resume_text', '').strip()
    uploaded_file = request.files.get('resume_file')
    extracted_text = ""

    if uploaded_file and uploaded_file.filename != '':
        filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(filename)
        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(filename)
        elif filename.lower().endswith('.txt'):
            with open(filename, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        else:
            extracted_text = ""
    elif resume_text:
        extracted_text = resume_text

    if extracted_text:
        clean_resume = clean_text(extracted_text)
        vect = tfidf.transform([clean_resume])
        pred_encoded = model.predict(vect)[0]
        prediction = le.inverse_transform([pred_encoded])[0]

    return render_template('result.html', prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)
