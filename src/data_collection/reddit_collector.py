import praw
import pandas as pd
from datetime import datetime
import os

class RedditCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        # Note: You'll need to register for Reddit API credentials
        self.reddit = praw.Reddit(
            client_id="your_client_id",
            client_secret="your_client_secret",
            user_agent="UG_Sentiment_Analysis_Bot"
        )
    
    def collect_ug_reddit_posts(self, subreddits, keywords, limit=1000):
        """Collect UG-related posts from Reddit"""
        posts_data = []
        
        target_subreddits = ["Ghana", "UniversityofGhana", "GhanaUniversities"]
        ug_keywords = ["university of ghana", "ug legon", "legon", "ug students"]
        
        for subreddit_name in target_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for UG-related posts
                for keyword in ug_keywords:
                    search_results = subreddit.search(keyword, limit=limit//len(ug_keywords)//len(target_subreddits))
                    
                    for post in search_results:
                        post_data = {
                            'id': post.id,
                            'title': post.title,
                            'content': post.selftext,
                            'score': post.score,
                            'created_utc': post.created_utc,
                            'num_comments': post.num_comments,
                            'subreddit': subreddit_name,
                            'keyword': keyword,
                            'source': 'reddit'
                        }
                        posts_data.append(post_data)
                        
            except Exception as e:
                print(f"Error collecting from r/{subreddit_name}: {e}")
                continue
        
        # Save data
        df = pd.DataFrame(posts_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"{self.output_dir}/reddit_raw_{timestamp}.csv", index=False)
        
        return df