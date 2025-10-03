import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import demoji
import string

class TextProcessor:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.english_stopwords = set(stopwords.words('english'))
        
        # Common Twi words (you'll need to expand this list)
        self.twi_words = {'me', 'wo', 'no', 'na', 'se', 'yi', 'ni', 'da', 'oo'}
        
    def clean_text(self, text):
        """Clean and normalize social media text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Replace user mentions
        text = re.sub(r'@\w+', 'USER', text)
        
        # Process hashtags (keep content, remove #)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Convert emojis to text descriptions
        text = demoji.replace_with_desc(text)
        
        # Normalize repeated characters (sooooo -> so)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def identify_code_mixing(self, text):
        """Identify English-Twi code-mixed content"""
        words = word_tokenize(text.lower())
        twi_count = sum(1 for word in words if word in self.twi_words)
        
        return {
            'is_code_mixed': twi_count > 0,
            'twi_ratio': twi_count / len(words) if words else 0,
            'total_words': len(words),
            'twi_words': twi_count
        }
    
    def preprocess_dataset(self, df, text_column='content'):
        """Preprocess entire dataset"""
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Identify code-mixing
        code_mixing_info = df['cleaned_text'].apply(self.identify_code_mixing)
        df['is_code_mixed'] = [info['is_code_mixed'] for info in code_mixing_info]
        df['twi_ratio'] = [info['twi_ratio'] for info in code_mixing_info]
        
        # Filter out very short texts
        df = df[df['cleaned_text'].str.len() > 10]
        
        return df

# Usage example
if __name__ == "__main__":
    processor = TextProcessor()
    
    # Load and process raw data
    df = pd.read_csv("data/raw/twitter_raw_20240101_120000.csv")
    processed_df = processor.preprocess_dataset(df)
    processed_df.to_csv("data/processed/twitter_processed.csv", index=False)