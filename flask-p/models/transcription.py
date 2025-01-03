import torch
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "openai/whisper-small.en"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

def transcribe_audio(audio_data):
    """Convert audio data to text using Hugging Face Whisper model."""
    try:
        input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.decode(predicted_ids[0])
        return transcription.strip()
    except Exception as e:
        raise RuntimeError(f"Error in transcription: {str(e)}")