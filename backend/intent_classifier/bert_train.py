import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from torch.optim import AdamW

# Sample multilingual training data
data = [
    # reset_password
    ("How do I reset my password?", "reset_password"),
    ("I forgot my password", "reset_password"),
    ("Ich habe mein Passwort vergessen", "reset_password"),
    ("mot de passe oublié", "reset_password"),
    ("Can't log in to my account", "reset_password"),
    ("Passwort zurücksetzen", "reset_password"),

    # track_order
    ("Where is my order?", "track_order"),
    ("Track my shipment", "track_order"),
    ("Wo ist meine Bestellung?", "track_order"),
    ("Suivre ma commande", "track_order"),
    ("Where can I find my package?", "track_order"),
    ("Bestellung verfolgen", "track_order"),

    # cancel_order
    ("I want to cancel my order", "cancel_order"),
    ("Cancel my subscription", "cancel_order"),
    ("Ich möchte meine Bestellung stornieren", "cancel_order"),
    ("Je veux annuler ma commande", "cancel_order"),
    ("Stop my membership", "cancel_order"),
    ("Abonnement kündigen", "cancel_order"),
]

df = pd.DataFrame(data, columns=["text", "label"])

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
joblib.dump(label_encoder, 'intent_classifier/label_encoder.joblib')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {**{k: v[idx] for k, v in self.encodings.items()}, 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

# Train/Validation Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)
train_dataset = IntentDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = IntentDataset(val_texts.tolist(), val_labels.tolist())

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=len(label_encoder.classes_)
)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} average loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch['labels'].tolist())
acc = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {acc:.4f}")

# Save everything
model.save_pretrained('intent_classifier/bert_model')
tokenizer.save_pretrained('intent_classifier/bert_tokenizer')
print("Model training complete and saved.")
