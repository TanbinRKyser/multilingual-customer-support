import torch
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F
from intent_classifier.bert_infer import model, tokenizer, label_encoder, device

explainer = LimeTextExplainer( class_names = list( label_encoder.classes_ ) )
model.to( device ).eval()

def predict_proba( texts ):
    enc = tokenizer( texts, return_tensors = "pt", padding = True, truncation = True)
    enc = { k: v.to( device ) for k, v in enc.items() }  

    with torch.no_grad():
        outputs = model( **enc )
        logits = outputs.logits
        probs = F.softmax( logits, dim = 1 ).detach().cpu().numpy()

    return probs

def explain_intent_lime( text: str, num_features: int = 8 ):
    exp = explainer.explain_instance( text, predict_proba, num_features = num_features )
    return exp.as_list()