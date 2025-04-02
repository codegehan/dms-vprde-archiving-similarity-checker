# extract_text.py
import os
import fitz
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# nltk.download("stopwords")
# nltk.download("punkt")  # For sentence tokenization
# stop_words = set(stopwords.words("english"))
stop_words = ENGLISH_STOP_WORDS

# Load the SentenceTransformer model for generating embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def extract_text_from_file(pdf_path, by_sentence=True):
    doc = fitz.open(pdf_path)
    extracted_data = []
    current_chapter = None
    in_references = False

    for page in doc:
        text = page.get_text()
        # Skip Table of Contents
        if re.search(r"\bTable of Contents?", text, re.IGNORECASE):
            current_chapter = None
        elif re.search(r"\bChapter\s+(1|I)\b", text, re.IGNORECASE):
            current_chapter = "I"
        elif re.search(r"\bChapter\s+(2|II)\b", text, re.IGNORECASE):
            current_chapter = "II"
        elif re.search(r"\bChapter\s+(3|III)\b", text, re.IGNORECASE):
            current_chapter = "III"
        elif re.search(r"\bChapter\s+(4|IV)\b", text, re.IGNORECASE):
            current_chapter = "IV"
        elif re.search(r"\bChapter\s+(5|V)\b", text, re.IGNORECASE):
            current_chapter = "V"
        
        if re.search(r"\bReferences\b", text, re.IGNORECASE):
            current_chapter = None
        
        # Extract content only for Chapter 1 to 5
        if current_chapter and not in_references:
            cleaned_text = preprocess_text(text)
            
            # Split by sentence or paragraph
            if by_sentence:
                sentences = sent_tokenize(cleaned_text)
                for sentence in sentences:
                    extracted_data.append({
                        "thesis_name": os.path.basename(pdf_path),
                        "content_extracted": sentence,
                        "chapter": current_chapter,
                        "extraction_type": "sentence"
                    })
            else:
                paragraphs = cleaned_text.split('\n\n')
                for paragraph in paragraphs:
                    extracted_data.append({
                        "thesis_name": os.path.basename(pdf_path),
                        "content_extracted": paragraph,
                        "chapter": current_chapter,
                        "extraction_type": "paragraph"
                    })

    return extracted_data

def start_training():
    if not os.path.exists("model"):
        os.makedirs("model")

    all_extracted_data = []
    files_folder = "files"
    for filename in os.listdir(files_folder):
        if filename.endswith(".pdf"):
            pdf_name = os.path.splitext(filename)[0]
            pdf_path = os.path.join(files_folder, filename)
            print(f"Processing {pdf_path}...")
            
            # Extract content by sentence or paragraph
            extracted_data = extract_text_from_file(pdf_path, by_sentence=True)
            all_extracted_data.extend(extracted_data)
    save_to_model_file(all_extracted_data, model_file="model/data_model.pkl")

def save_to_model_file(extracted_data, model_file="model/data_model.pkl"):
    # Generate embeddings for all sentences
    sentences = [item["content_extracted"] for item in extracted_data]
    embeddings = model.encode(sentences)

    # Combine data and embeddings into a single object
    model_data = {
        "thesis_data": extracted_data,
        "embeddings": embeddings
    }

    # Save the model data to a file
    with open(model_file, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {model_file}.")

if __name__ == "__main__":
    start_training()
    print("All PDFs processed and model saved!")