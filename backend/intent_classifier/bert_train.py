import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from torch.optim import AdamW 

# Data
data = [
    ("How do I reset my password?", "reset_password"),
    ("I forgot my password", "reset_password"),
    ("Where is my order?", "track_order"),
    ("Track my shipment", "track_order"),
    ("I want to cancel my order", "cancel_order"),
    ("Cancel my subscription", "cancel_order"),
]

df = pd.DataFrame(data, columns=["text", "label"])


# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
joblib.dump(label_encoder, 'intent_classifier/label_encoder.joblib')

## Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class IntentDataset( Dataset ):
    def __init__( self, texts, labels ):
        self.encodings = tokenizer( texts, padding=True, truncation=True, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__( self, idx ):
        return {**{k: v[idx] for k, v in self.encodings.items()}, 'labels': self.labels[idx]}

    def __len__( self ):
        return len(self.labels)

dataset = IntentDataset( df['text'].tolist(), df['label'].tolist() )
dataloader = DataLoader( dataset, batch_size=2, shuffle=True )


## Model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_encoder.classes_))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

## training loop
for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        # inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} loss: {loss.item()}")

# Save the model
model.save_pretrained('intent_classifier/bert_model')
tokenizer.save_pretrained('intent_classifier/bert_tokenizer')
print("Model training complete and saved.")