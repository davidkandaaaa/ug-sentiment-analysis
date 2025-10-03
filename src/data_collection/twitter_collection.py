import snscrape.modules.twitter as sntwitter
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os

class TwitterCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_ug_tweets(self, keywords, start_date, end_date, max_tweets=5000):
        """Collect UG-related tweets"""
        tweets_data = []
        
        # UG-specific keywords
        ug_keywords = [
            "University of Ghana",
            "UG Legon",
            "#UGLegon",
            "#UGLife", 
            "Legon campus",
            "@UniversityGhana"
        ]
        
        for keyword in ug_keywords:
            print(f"Collecting tweets for: {keyword}")
            
            query = f'"{keyword}" since:{start_date} until:{end_date} lang:en'
            
            try:
                tweet_count = 0
                for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                    if tweet_count >= max_tweets // len(ug_keywords):
                        break
                    
                    tweet_data = {
                        'id': tweet.id,
                        'date': tweet.date,
                        'content': tweet.content,
                        'username': tweet.user.username,
                        'likes': tweet.likeCount,
                        'retweets': tweet.retweetCount,
                        'replies': tweet.replyCount,
                        'keyword': keyword,
                        'source': 'twitter'
                    }
                    tweets_data.append(tweet_data)
                    tweet_count += 1
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error collecting tweets for {keyword}: {e}")
                continue
        
        # Save raw data
        df = pd.DataFrame(tweets_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"{self.output_dir}/twitter_raw_{timestamp}.csv", index=False)
        
        print(f"Collected {len(tweets_data)} tweets")
        return df

# Usage example
if __name__ == "__main__":
    collector = TwitterCollector()
    tweets = collector.collect_ug_tweets(
        keywords=[], 
        start_date="2024-01-01", 
        end_date="2024-06-30"
    )