import os
import tempfile
from flask import Flask, request
from flask_cors import CORS

from services.similarity_service import SimilarityService
from utils.response_handler import ResponseHandler

app = Flask(__name__)
CORS(app)

similarity_service = SimilarityService()

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return ResponseHandler.error("No file part", 400)
        
        file = request.files['file']
        if file.filename == '':
            return ResponseHandler.error('No selected file', 400)
        
        if not file.filename.endswith('.pdf'):
            return ResponseHandler.error("Invalid file format. Please upload a PDF file.", 400)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        file.save(temp_file.name)
        temp_file.close()

        try:
            result = similarity_service.analyze_thesis(temp_file.name)

            os.unlink(temp_file.name)

            return ResponseHandler.success(result, 'File analyzed successfully')
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

            return ResponseHandler.handle_exception(e)
    except Exception as e:
        return ResponseHandler.handle_exception(e)
    
if __name__ == '__main__':
    app.run(debug=True, port=5566)
    print('API Service running on port: 5566')