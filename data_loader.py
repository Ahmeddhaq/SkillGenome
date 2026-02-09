import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class SkillDataLoader:
    def __init__(self, n_users=100):
        secret_token = None
        secret_so_key = None
        try:
            import streamlit as st
            secret_token = st.secrets.get("GITHUB_API")
            secret_so_key = st.secrets.get("STACKOVERFLOW_KEY")
        except Exception:
            pass

        self.token = secret_token or os.getenv("GITHUB_API")
        self.so_key = secret_so_key or os.getenv("STACKOVERFLOW_KEY")
        self.headers = {"Authorization": f"token {self.token}"} if self.token else {}
        self.base_url = "https://api.github.com"
        self.so_base_url = "https://api.stackexchange.com/2.3"
        self.n_users = n_users
    
    def fetch_users(self, location="India"):
        """Fetch GitHub users from specified location"""
        users = []
        page = 1
        
        while len(users) < self.n_users:
            query = f"location:{location} followers:>10 type:user"
            url = f"{self.base_url}/search/users?q={query}&per_page=100&page={page}"
            
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise ValueError(f"GitHub API error {resp.status_code}: {err}")
            
            data = resp.json().get("items", [])
            if not data:
                break
            
            for user in data:
                if len(users) >= self.n_users:
                    break
                
                # Fetch full user details to get followers and location
                user_detail_url = f"{self.base_url}/users/{user['login']}"
                detail_resp = requests.get(user_detail_url, headers=self.headers)
                
                if detail_resp.status_code == 200:
                    user_data = detail_resp.json()
                    users.append({
                        "login": user_data["login"],
                        "followers": user_data.get("followers", 0),
                        "public_repos": user_data.get("public_repos", 0),
                        "avatar_url": user_data.get("avatar_url", ""),
                        "location": user_data.get("location", "")
                    })
            
            page += 1
        
        return users[:self.n_users]
    
    def fetch_user_repos(self, username):
        """Fetch user's repositories and calculate metrics"""
        url = f"{self.base_url}/users/{username}/repos?per_page=100"
        resp = requests.get(url, headers=self.headers)
        
        if resp.status_code != 200:
            return {"stars": 0, "languages": [], "commits": 0}
        
        repos = resp.json()
        stars = sum(r["stargazers_count"] for r in repos)
        languages = list(set([r["language"] for r in repos if r["language"]]))
        commits = len(repos) * 15
        
        return {"stars": stars, "languages": languages, "commits": commits}
    
    def fetch_stackoverflow_data(self, username):
        """Fetch Stack Overflow data for a user"""
        try:
            # Search for user by display name
            search_url = f"{self.so_base_url}/users"
            params = {
                "inname": username,
                "site": "stackoverflow",
                "key": self.so_key,
                "pagesize": 1
            }
            
            resp = requests.get(search_url, params=params)
            
            if resp.status_code != 200 or not resp.json().get("items"):
                return {"reputation": 0, "answers": 0, "questions": 0, "badges": 0}
            
            user_data = resp.json()["items"][0]
            
            return {
                "reputation": user_data.get("reputation", 0),
                "answers": user_data.get("answer_count", 0),
                "questions": user_data.get("question_count", 0),
                "badges": user_data.get("badge_counts", {}).get("gold", 0) * 3 + 
                         user_data.get("badge_counts", {}).get("silver", 0) * 2 + 
                         user_data.get("badge_counts", {}).get("bronze", 0)
            }
        except Exception as e:
            return {"reputation": 0, "answers": 0, "questions": 0, "badges": 0}
    
    def get_region(self, location_str):
        """Map GitHub location to Indian region"""
        if not location_str:
            return "Central"
        
        location = location_str.lower()
        
        # North India
        north_cities = ['delhi', 'new delhi', 'punjab', 'chandigarh', 'himachal', 'haryana', 'noida', 'gurugram', 'jaipur', 'rajasthan']
        # South India
        south_cities = ['bangalore', 'bengaluru', 'hyderabad', 'chennai', 'pune', 'telangana', 'karnataka', 'tamil nadu', 'andhra', 'kerala', 'cochin']
        # East India
        east_cities = ['kolkata', 'west bengal', 'bihar', 'odisha', 'jharkhand', 'guwahati', 'assam', 'patna']
        # West India
        west_cities = ['mumbai', 'pune', 'ahmedabad', 'rajasthan', 'goa', 'surat', 'vadodara', 'indore']
        # Central India
        central_cities = ['bhopal', 'madhya pradesh', 'chhattisgarh', 'lucknow', 'uttar pradesh']
        
        if any(city in location for city in north_cities):
            return "North"
        elif any(city in location for city in south_cities):
            return "South"
        elif any(city in location for city in east_cities):
            return "East"
        elif any(city in location for city in west_cities):
            return "West"
        elif any(city in location for city in central_cities):
            return "Central"
        
        # Default fallback
        return "Central"
    
    def build_features(self, users):
        """Convert GitHub + Stack Overflow data into skill features"""
        data = []
        
        for i, user in enumerate(users):
            repo_data = self.fetch_user_repos(user["login"])
            so_data = self.fetch_stackoverflow_data(user["login"])
            
            features = {
                "id": i,
                "username": user["login"],
                # GitHub features
                "followers": max(1, user["followers"]),
                "public_repos": max(1, user["public_repos"]),
                "stars": max(1, repo_data["stars"]),
                "commits": max(1, repo_data["commits"]),
                "languages": len(repo_data["languages"]),
                # Stack Overflow features
                "so_reputation": max(1, so_data["reputation"]),
                "so_answers": so_data["answers"],
                "so_questions": so_data["questions"],
                "so_badges": so_data["badges"],
                # Geographic
                "region": self.get_region(user.get("location", "")),
                "lat": np.random.uniform(8, 37),
                "lon": np.random.uniform(68, 97),
                "urban": 1
            }
            
            data.append(features)
            
            if (i + 1) % 50 == 0:
                print(f"  Fetched {i + 1}/{self.n_users} users...")
        
        return pd.DataFrame(data)
    
    def detect_adversarial(self, df):
        """Detect suspicious/fake signals using rule-based heuristics"""
        print("Running adversarial detection...")
        
        suspicious_flags = []
        trust_scores = []
        
        for idx, row in df.iterrows():
            flags = []
            trust = 100  # Start with perfect score
            
            # Rule 1: High followers but zero activity (bot accounts)
            if row['followers'] > 100 and row['public_repos'] == 1:
                flags.append("inactive_influencer")
                trust -= 30
            
            # Rule 2: Impossible star/repo ratio (fake stars)
            if row['stars'] > row['public_repos'] * 200:
                flags.append("star_manipulation")
                trust -= 25
            
            # Rule 3: Zero engagement (dormant account)
            if row['commits'] == row['public_repos'] * 15 and row['stars'] == 1:
                flags.append("zero_engagement")
                trust -= 20
            
            # Rule 4: SO reputation without answers (bought account)
            if row['so_reputation'] > 1000 and row['so_answers'] == 0:
                flags.append("so_account_suspicious")
                trust -= 25
            
            # Rule 5: Perfect round numbers (scripted/fake data)
            if (row['followers'] % 100 == 0 and row['followers'] > 100 and 
                row['stars'] % 100 == 0 and row['stars'] > 100):
                flags.append("rounded_metrics")
                trust -= 15
            
            # Rule 6: Extreme outliers (statistical anomaly)
            if row['followers'] > df['followers'].quantile(0.99) * 3:
                flags.append("extreme_outlier")
                trust -= 10
            
            suspicious_flags.append(",".join(flags) if flags else "clean")
            trust_scores.append(max(0, trust))
        
        df['suspicious_flags'] = suspicious_flags
        df['trust_score'] = trust_scores
        df['is_suspicious'] = df['trust_score'] < 70
        
        suspicious_count = df['is_suspicious'].sum()
        print(f"✓ Adversarial detection complete: {suspicious_count}/{len(df)} flagged as suspicious")
        
        return df
    
    def load_all(self):
        """Main entry: fetch GitHub + Stack Overflow users and build feature matrix"""
        if not self.token:
            raise ValueError("Missing GITHUB_API. Set it in Streamlit secrets or .env.")
        print(f"Fetching {self.n_users} users from multiple sources (GitHub + Stack Overflow)...")
        users = self.fetch_users()
        
        if not users:
            raise ValueError("No users found. Check GITHUB_API, rate limits, or search query filters.")
        
        print(f"✓ Fetched {len(users)} users\nBuilding multi-platform feature matrix...")
        df = self.build_features(users)
        
        # Apply adversarial detection
        df = self.detect_adversarial(df)
        
        print(f"✓ Built {len(df)} profiles with {len(df.columns)} features from 2 platforms")
        return df
