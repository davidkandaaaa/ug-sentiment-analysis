import pandas as pd
import streamlit as st
import json
from datetime import datetime

class AnnotationTool:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = pd.read_csv(data_file)
        self.annotations_file = "data/annotated/annotations.json"
        self.load_existing_annotations()
    
    def load_existing_annotations(self):
        """Load existing annotations if available"""
        try:
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
        except FileNotFoundError:
            self.annotations = {}
    
    def save_annotation(self, post_id, sentiment, annotator):
        """Save annotation to file"""
        self.annotations[str(post_id)] = {
            'sentiment': sentiment,
            'annotator': annotator,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def create_streamlit_interface(self):
        """Create Streamlit interface for annotation"""
        st.title("UG Student Sentiment Annotation Tool")
        
        # Annotator identification
        annotator_name = st.text_input("Your name:")
        
        if annotator_name:
            # Get unannotated posts
            unannotated_posts = self.df[~self.df.index.isin([int(k) for k in self.annotations.keys()])]
            
            if len(unannotated_posts) > 0:
                # Current post
                current_post = unannotated_posts.iloc[0]
                
                st.write(f"**Post {current_post.name + 1} of {len(self.df)}**")
                st.write(f"**Source:** {current_post.get('source', 'Unknown')}")
                st.write(f"**Date:** {current_post.get('date', 'Unknown')}")
                st.write("---")
                st.write(f"**Content:** {current_post['cleaned_text']}")
                
                # Sentiment selection
                sentiment = st.radio(
                    "Select sentiment:",
                    ["Positive", "Negative", "Neutral"],
                    key=f"sentiment_{current_post.name}"
                )
                
                # Submit annotation
                if st.button("Submit Annotation"):
                    self.save_annotation(current_post.name, sentiment.lower(), annotator_name)
                    st.success("Annotation saved!")
                    st.experimental_rerun()
            
            else:
                st.success("All posts have been annotated!")
                
                # Show annotation statistics
                st.write("**Annotation Statistics:**")
                sentiments = [ann['sentiment'] for ann in self.annotations.values()]
                sentiment_counts = pd.Series(sentiments).value_counts()
                st.bar_chart(sentiment_counts)

# Run annotation tool
if __name__ == "__main__":
    tool = AnnotationTool("data/processed/combined_processed.csv")
    tool.create_streamlit_interface()