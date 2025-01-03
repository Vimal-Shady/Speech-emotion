from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    """Analyze sentiment of a given text."""
    try:
        return analyzer(text)[0]
    except Exception as e:
        raise RuntimeError(f"Error in sentiment analysis: {str(e)}")