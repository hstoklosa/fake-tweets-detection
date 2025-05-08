import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  accuracy_score, 
  f1_score,
  classification_report,
  confusion_matrix,
  ConfusionMatrixDisplay,  
)


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def sample_dataset(df, stratify_col, train_size=0.50, random_state=42):
    _, new_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[stratify_col],
        random_state=random_state
    )

    return new_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}


def plot_confusion_matrix(y_test, predicted_labels):
  cm = confusion_matrix(y_test, predicted_labels)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot()


def export_model(model, tokenizer, path):
    if not os.path.exists(path):
        os.makedirs(path)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
