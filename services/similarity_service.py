from model_run import extract_new_thesis, generate_embeddings, load_model_file, create_faiss_index, find_similar_sentences, calculate_overall_similarity, determine_status

class SimilarityService:
    def analyze_thesis(self, file_path):
        # Load the model data
        model_data = load_model_file()
        
        # Check if model_data has the expected structure
        if not isinstance(model_data, dict):
            raise ValueError(f"Model data is not a dictionary. Got {type(model_data)}")
        
        if "embeddings" not in model_data:
            raise ValueError("Model data missing 'embeddings' key")
        
        # Use 'thesis_data' instead of 'processed_theses'
        if "thesis_data" not in model_data:
            raise ValueError("Model data missing 'thesis_data' key")
        
        # Extract text from the new thesis
        new_thesis_data = extract_new_thesis(file_path)
        new_sentences = [item["content_extracted"] for item in new_thesis_data]
        
        if not new_sentences:
            raise ValueError('No text could be extracted from the PDF')
        
        # Generate embeddings for the new thesis
        new_embeddings = generate_embeddings(new_sentences)
        
        # Create FAISS index from the stored embeddings
        index = create_faiss_index(model_data["embeddings"])
        
        # Find similar sentences
        similar_sentences = find_similar_sentences(
            new_embeddings, 
            index, 
            new_sentences, 
            model_data["thesis_data"]  # Use 'thesis_data' instead of 'processed_theses'
        )
        
        # Calculate overall similarity
        overall_similarity = calculate_overall_similarity(new_embeddings, index)
        
        # Determine status (0 for high similarity, 1 for acceptable)
        status = determine_status(overall_similarity)

        return {
            'status': status,
            'overall_similarity': overall_similarity,
            'similar_sentences': similar_sentences
        }