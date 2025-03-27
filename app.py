from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import textwrap
import PyPDF2
import io
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Load model and tokenizer with error handling
try:
    model_path = "./summarization_model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory '{model_path}' not found. Please download a pre-trained model.")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise  

def process_text(text, max_input_length=10000):
    """Process text in chunks and generate summaries."""
    if len(text) > max_input_length:
        text = text[:max_input_length]  
    
    # Split text into chunks of ~1000 characters
    chunks = textwrap.wrap(text, width=1000, break_long_words=False)
    summaries = []
    
    for chunk in chunks:
        try:
            inputs = tokenizer(
                chunk,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            summaries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        except RuntimeError as e:
            print(f"Error processing chunk: {e}. Falling back to CPU.")
            model.to("cpu")  # Fallback to CPU if GPU fails
            inputs = inputs.to("cpu")
            outputs = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summaries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            model.to(device)  # Move back to GPU
    
    return " ".join(summaries)

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        return text
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        text = ""
        if 'text' in request.form and request.form['text'].strip():
            text = request.form['text'].strip()
        elif 'file' in request.files:
            file = request.files['file']
            if not file.filename:
                return jsonify({'error': 'No file uploaded'})
            if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({'error': 'File size exceeds 5MB limit'})
            if file.filename.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
            else:
                return jsonify({'error': 'Unsupported file type. Use .pdf or .txt'})
        
        if not text or len(text) < 100:
            return jsonify({'error': 'Input must be at least 100 characters'})
        
        summary = process_text(text)
        return jsonify({'summary': summary})
    
    except ValueError as ve:
        return jsonify({'error': str(ve)})
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)