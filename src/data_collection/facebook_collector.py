import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class FacebookCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        self.setup_driver()
    
    def setup_driver(self):
        """Setup Chrome driver for Facebook scraping"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def collect_ug_facebook_posts(self, page_urls, max_posts=1000):
        """Collect posts from UG Facebook pages"""
        posts_data = []
        
        ug_pages = [
            "https://www.facebook.com/UniversityofGhanaOfficial",
            # Add other UG-related Facebook pages
        ]
        
        for page_url in ug_pages:
            print(f"Collecting from: {page_url}")
            
            try:
                self.driver.get(page_url)
                time.sleep(3)
                
                # Scroll and collect posts (simplified approach)
                posts = self.driver.find_elements("css selector", "[data-pagelet='FeedUnit']")
                
                for post in posts[:max_posts//len(ug_pages)]:
                    try:
                        content = post.find_element("css selector", "[data-testid='post_message']").text
                        timestamp = post.find_element("css selector", "time").get_attribute("datetime")
                        
                        post_data = {
                            'content': content,
                            'timestamp': timestamp,
                            'source': 'facebook',
                            'page': page_url
                        }
                        posts_data.append(post_data)
                        
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Error collecting from {page_url}: {e}")
                continue
        
        # Save data
        df = pd.DataFrame(posts_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"{self.output_dir}/facebook_raw_{timestamp}.csv", index=False)
        
        self.driver.quit()
        return df