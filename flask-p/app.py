import os
from flask import Flask, render_template, request, jsonify
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import soundfile as sf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Whisper model for transcription
model_name = "openai/whisper-small.en"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load sentiment analysis model using PyTorch
sentiment_analyzer = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment", framework='pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Read the audio file using soundfile
    try:
        # Load the audio file
        audio_input, sampling_rate = sf.read(file_path)
        
        # Process the audio input
        input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features
    except Exception as e:
        return jsonify({'error': f'Error processing audio file: {str(e)}'})

    # Perform transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0])
    transcription = transcription.replace("<|startoftranscript|>", "").replace("<|notimestamps|>", "").strip()

    # Perform sentiment analysis
    sentiment = sentiment_analyzer(transcription)[0]

    return jsonify({
        'transcription': transcription,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)