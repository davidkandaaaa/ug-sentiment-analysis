"""
Advanced Model Training Coordinator
File: src/models/advanced_trainer.py
"""

import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime
import logging

from .bert_model import BERTSentimentClassifier
from .bilstm_attention import BiLSTMWithAttention

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedModelTrainer:
    """Coordinates training of advanced models (BERT and Bi-LSTM)"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.results = {}
        
        # Create output directories
        self._create_directories()
    
    def _default_config(self):
        """Default configuration for training"""
        return {
            'bert': {
                'epochs': 3,
                'batch_size': 16,
                'model_name': 'bert-base-uncased'
            },
            'bilstm': {
                'epochs': 10,
                'batch_size': 32,
                'vocab_size': 10000,
                'embedding_dim': 300,
                'max_length': 100,
                'lstm_units': 128
            },
            'data': {
                'text_column': 'cleaned_text',
                'target_column': 'sentiment'
            },
            'output': {
                'models_dir': 'models',
                'results_dir': 'results'
            }
        }
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            f"{self.config['output']['models_dir']}/bert",
            f"{self.config['output']['models_dir']}/bilstm_attention",
            f"{self.config['output']['results_dir']}/week5",
            "logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate_data(self, df):
        """Validate input data format"""
        required_columns = [
            self.config['data']['text_column'],
            self.config['data']['target_column