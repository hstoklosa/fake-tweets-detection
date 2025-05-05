import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
  accuracy_score, 
  f1_score,
  classification_report,
  confusion_matrix,
  ConfusionMatrixDisplay,  
)
from tqdm import tqdm


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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


def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, device):
    """Generic training function for transformer models"""
    best_val_accuracy, best_model_state = 0, None
    training_stats = [] # for the training loop
    
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1} / {epochs} ========")
        print("Training...")
        
        # Reset total loss at the start of each epoch
        total_train_loss = 0
        model.train()
        
        train_progress_bar = tqdm(train_dataloader, desc="Training", leave=True)
        
        for batch in train_progress_bar:
            # Clear gradients
            model.zero_grad()
            
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0) # prevent exploding gradients
            optimizer.step() # update weights
            scheduler.step()  # update learning rate
            
            train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        print("Running validation...")
        model.eval()
        
        predictions = []
        true_labels = []
        total_eval_loss = 0
        
        for batch in tqdm(val_dataloader, desc="Validation", leave=True):
            # Get data from the current batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            total_eval_loss += loss.item()
            
            # Get predictions made by the model
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            label_ids = labels.cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(label_ids)
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(true_labels, predictions)
        val_f1 = f1_score(true_labels, predictions)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        
        # Store stats
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        }
        training_stats.append(epoch_stats)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"New best model with validation accuracy: {best_val_accuracy:.4f}")
    
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
    
    return model, training_stats


def evaluate_model(model, dataloader, device):
    """Evaluate model performance on a dataset"""
    model.eval()
    
    predictions, true_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            label_ids = labels.cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(label_ids)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }


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
