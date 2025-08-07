import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import logging

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

## paths
model_path = 'intent_classifier/bert_model'
tokenizer_path = 'intent_classifier/bert_tokenizer'
label_encoder_path = 'intent_classifier/label_encoder.joblib'


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model.to(device)
model.eval()

# Load label encoder
label_encoder = joblib.load('intent_classifier/label_encoder.joblib')

# Function to predict intent
def predict_intent(texts):

    if isinstance(texts, str):
        texts = [texts]
    
    inputs = tokenizer( 
        texts, 
        return_tensors='pt', 
        padding=True, 
        truncation=True ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
    
    labels = label_encoder.inverse_transform(predictions)
    confidences = [probs[i, pred].item() for i, pred in enumerate(predictions)]

    return list(zip(labels, confidences))