import sys
import os

import numpy
import pickle
import pandas as pd

relative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grouped_posts.pkl")
with open(relative_path, 'rb') as f:  # or use file_path
    grouped_posts = pickle.load(f)

df_posts = pd.DataFrame(grouped_posts)  # Critical fix here

min_extremity = df_posts['extremity'].min()
max_extremity = df_posts['extremity'].max()

# Avoid division by zero if all values are the same
if max_extremity != min_extremity:
    # Normalize using min-max scaling
    df_posts['extremity'] = (df_posts['extremity'] - min_extremity) / (max_extremity - min_extremity)

# Check columns
if 'extremity' not in df_posts.columns or 'label' not in df_posts.columns:
    print("Error: Missing required columns")
    exit()

# Calculate statistics
statistics = df_posts.groupby('label')['extremity'].agg(['mean', 'min', 'max', 'median'])
print("\nExtremity Statistics Grouped by Label:")
print(statistics)
statistics.to_csv('extremity_statistics.csv', index=True)
total_posts = len(df_posts)
print(f"Total number of posts: {total_posts}")
# Print first 100 posts
#print("\nFirst 100 Posts:")
#for i, post in enumerate(grouped_posts[:100]):  # Direct list access
    #print(f"\nPost {i+1}:")
    #print(f"content: {post['content']}")
    #print(f"label: {post['label']}")
    #print(f"extremity: {post['extremity']}")
    #print(f"tc: {post['tc']}")
    #print("-" * 50)