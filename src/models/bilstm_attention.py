import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, 
    Bidirectional, GlobalMaxPool1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)
    
    def call(self, encoder_outputs):
        # encoder_outputs shape: (batch_size, seq_len, hidden_size)
        score = tf.nn.tanh(self.W(encoder_outputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class BiLSTMWithAttention:
    def __init__(self, vocab_size=10000, embedding_dim=300, max_length=100, lstm_units=128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.tokenizer = None
        self.model = None
        self.label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    def build_model(self):
        """Build Bi-LSTM with Attention model"""
        # Input layer
        input_layer = Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            trainable=True
        )(input_layer)
        
        # Bidirectional LSTM
        lstm_out = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(embedding)
        
        # Attention mechanism
        attention_layer = AttentionLayer(self.lstm_units * 2)
        context_vector, attention_weights = attention_layer(lstm_out)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(context_vector)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        
        # Output layer
        output = Dense(3, activation='softmax')(dropout2)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def prepare_data(self, df, text_column='cleaned_text', target_column='sentiment'):
        """Prepare data for training"""
        texts = df[text_column].fillna('').astype(str).tolist()
        labels = df[target_column].map(self.label_to_id).tolist()
        
        # Tokenization
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Convert labels to categorical
        y = to_categorical(labels, num_classes=3)
        
        return X, y
    
    def train_model(self, df, validation_split=0.2, epochs=10, batch_size=32):
        """Train the Bi-LSTM with Attention model"""
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Print model summary
        print("Model Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        # Train model
        print("Training Bi-LSTM with Attention model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.3f}")
        
        return history, test_accuracy
    
    def save_model(self, model_path="models/bilstm_attention"):
        """Save model and tokenizer"""
        import os
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        self.model.save(f"{model_path}/model.h5")
        
        # Save tokenizer
        with open(f"{model_path}/tokenizer.pickle", "wb") as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model and tokenizer saved to {model_path}")
    
    def load_model(self, model_path="models/bilstm_attention"):
        """Load saved model and tokenizer"""
        # Load model
        self.model = tf.keras.models.load_model(
            f"{model_path}/model.h5",
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        
        # Load tokenizer
        with open(f"{model_path}/tokenizer.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
        
        print(f"Model and tokenizer loaded from {model_path}")
    
    def predict(self, texts, model_path=None):
        """Make predictions on new texts"""
        # Load model and tokenizer if not in memory
        if self.model is None and model_path:
            self.load_model(model_path)
        
        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Train model first or provide model_path.")
        
        # Preprocess texts
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Convert back to labels
        id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_labels = [id_to_label[pred] for pred in predicted_classes]
        
        return predicted_labels, confidence_scores
    
    def predict_with_attention(self, texts, model_path=None):
        """Make predictions and return attention weights"""
        if self.model is None and model_path:
            self.load_model(model_path)
        
        # Create a model that outputs both predictions and attention weights
        attention_model = Model(
            inputs=self.model.input,
            outputs=[self.model.output, self.model.get_layer('attention_layer').output[1]]
        )
        
        # Preprocess texts
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Predict with attention
        predictions, attention_weights = attention_model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Convert back to labels
        id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_labels = [id_to_label[pred] for pred in predicted_classes]
        
        return predicted_labels, attention_weights

# Training script
def train_bilstm_model():
    """Main training function"""
    # Load data
    try:
        df = pd.read_csv("data/annotated/annotated_posts.csv")
        print(f"Loaded {len(df)} annotated posts")
    except FileNotFoundError:
        print("Error: Could not find annotated data file.")
        print("Please ensure you have annotated data at 'data/annotated/annotated_posts.csv'")
        return
    
    # Check data format
    required_columns = ['cleaned_text', 'sentiment']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in data")
            return
    
    # Check sentiment labels
    valid_sentiments = {'positive', 'negative', 'neutral'}
    unique_sentiments = set(df['sentiment'].unique())
    if not unique_sentiments.issubset(valid_sentiments):
        print(f"Warning: Found unexpected sentiment labels: {unique_sentiments - valid_sentiments}")
    
    print(f"Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Initialize and train model
    bilstm_model = BiLSTMWithAttention(
        vocab_size=10000,
        embedding_dim=300,
        max_length=100,
        lstm_units=128
    )
    
    # Train model
    history, accuracy = bilstm_model.train_model(
        df, 
        epochs=10,
        batch_size=32
    )
    
    # Save model
    bilstm_model.save_model("models/bilstm_attention")
    
    print(f"Bi-LSTM Training completed. Test accuracy: {accuracy:.3f}")
    
    # Test prediction
    sample_texts = [
        "I love the new facilities at UG Legon!",
        "Registration system is so frustrating every semester",
        "Library opens at 8am tomorrow"
    ]
    
    predictions, confidence = bilstm_model.predict(sample_texts)
    
    print("\nSample Predictions:")
    for text, pred, conf in zip(sample_texts, predictions, confidence):
        print(f"Text: {text}")
        print(f"Predicted sentiment: {pred} (confidence: {conf:.3f})")
        print("-" * 50)
    
    return bilstm_model, history

# Usage example
if __name__ == "__main__":
    # Train the model
    model, history = train_bilstm_model()
    
    # Plot training history if available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/bilstm_attention/training_history.png')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping training plots.")
    except Exception as e:
        print(f"Could not create training plots: {e}")