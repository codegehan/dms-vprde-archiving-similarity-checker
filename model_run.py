# check_similarity.py
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from model_train import extract_text_from_file

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_model_file(model_file="model/data_model.pkl"):
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)
    return model_data

def extract_new_thesis(pdf_path):
    # Extract and preprocess new thesis
    extracted_data = extract_text_from_file(pdf_path, by_sentence=True)
    return extracted_data

def generate_embeddings(new_sentences, device="cpu"):
    model.to(device) # Use GPU if Available
    return model.encode(new_sentences, device=device)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings.astype('float32'))  # Add embeddings to the index
    return index

def find_similar_sentences(new_embeddings, index, new_sentences, processed_theses, threshold=0.8):
    results = []
    
    for i, new_embedding in enumerate(new_embeddings):
        # Search the FAISS index for similar embeddings
        distances, indices = index.search(np.array([new_embedding]).astype('float32'), k=5)
        
        for idx, distance in zip(indices[0], distances[0]):
            similarity = 1 - distance  # Convert L2 distance to similarity
            if similarity > threshold:
                results.append({
                    "new_sentence": new_sentences[i],
                    "similar_title": processed_theses[idx]["thesis_name"].replace('.pdf', '').strip(),
                    "similar_chapter": processed_theses[idx]["chapter"],
                    "similar_sentence": processed_theses[idx]["content_extracted"],
                    "similarity_score": float(similarity)
                })
    
    return results

def calculate_overall_similarity(new_embeddings, index, threshold=0.8):
    high_similarity_count = 0
    
    for new_embedding in new_embeddings:
        distances, _ = index.search(np.array([new_embedding]).astype('float32'), k=1)
        similarity = 1 - distances[0][0]  # Convert L2 distance to similarity
        if similarity > threshold:
            high_similarity_count += 1
    
    return high_similarity_count / len(new_embeddings)

def determine_status(overall_similarity, threshold=0.8):
    return 0 if overall_similarity > threshold else 1