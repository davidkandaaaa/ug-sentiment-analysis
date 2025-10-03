import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import joblib

class BaselineModels:
    def __init__(self):
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'svm': LinearSVC(max_iter=1000)
        }
        self.vectorizers = {}
        self.fitted_models = {}
    
    def prepare_data(self, df, text_column='cleaned_text', target_column='sentiment'):
        """Prepare data for training"""
        X = df[text_column].fillna('')
        y = df[target_column]
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def create_features(self, X_train, X_test, method='tfidf'):
        """Create text features"""
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:  # count
            vectorizer = CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        self.vectorizers[method] = vectorizer
        
        return X_train_vec, X_test_vec
    
    def train_baseline_models(self, df):
        """Train all baseline models"""
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Create features
        X_train_tfidf, X_test_tfidf = self.create_features(X_train, X_test, 'tfidf')
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train_tfidf, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_tfidf)
            
            # Evaluate
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[model_name] = {
                'accuracy': report['accuracy'],
                'classification_report': report,
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            # Save model
            self.fitted_models[model_name] = model
            
            print(f"{model_name} accuracy: {report['accuracy']:.3f}")
        
        return results
    
    def save_models(self, output_dir="models/baseline"):
        """Save trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.fitted_models.items():
            joblib.dump(model, f"{output_dir}/{model_name}.pkl")
        
        for vec_name, vectorizer in self.vectorizers.items():
            joblib.dump(vectorizer, f"{output_dir}/vectorizer_{vec_name}.pkl")

# Usage example
if __name__ == "__main__":
    # Load annotated data
    df = pd.read_csv("data/annotated/annotated_posts.csv")
    
    # Train baseline models
    baseline = BaselineModels()
    results = baseline.train_baseline_models(df)
    baseline.save_models()
    
    # Print results
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {result['accuracy']:.3f}")0