import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def prepare_data(self, df, text_column='cleaned_text', target_column='sentiment'):
        """Prepare data for BERT training"""
        # Convert string labels to numeric
        df['label_numeric'] = df[target_column].map(self.label_to_id)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_column], df['label_numeric'], 
            test_size=0.2, random_state=42, stratify=df['label_numeric']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.25, random_state=42, stratify=y_train
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Create PyTorch datasets"""
        train_dataset = SentimentDataset(X_train, y_train, self.tokenizer)
        val_dataset = SentimentDataset(X_val, y_val, self.tokenizer)
        test_dataset = SentimentDataset(X_test, y_test, self.tokenizer)
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, df, output_dir="models/bert", epochs=3):
        """Train BERT model"""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
        train_dataset, val_dataset, test_dataset = self.create_datasets(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            seed=42
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        print("Starting BERT training...")
        trainer.train()
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_dataset)
        print(f"Test Results: {test_results}")
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer, test_results
    
    def predict(self, texts, model_path="models/bert"):
        """Make predictions on new texts"""
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        predictions = []
        model.eval()
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=128
                )
                
                outputs = model(**inputs)
                predicted_class = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(self.id_to_label[predicted_class])
        
        return predictions

# Usage example
if __name__ == "__main__":
    # Load annotated data
    df = pd.read_csv("data/annotated/annotated_posts.csv")
    
    # Train BERT model
    bert_classifier = BERTSentimentClassifier()
    trainer, results = bert_classifier.train_model(df)
    
    print(f"BERT Training completed. Test accuracy: {results['eval_accuracy']:.3f}")
    
    # Example prediction
    sample_texts = [
        "I love the new facilities at UG Legon!",
        "Registration system is so frustrating every semester",
        "Library opens at 8am tomorrow"
    ]
    
    predictions = bert_classifier.predict(sample_texts)
    
    for text, pred in zip(sample_texts, predictions):
        print(f"Text: {text}")
        print(f"Predicted sentiment: {pred}")
        print("-" * 50)