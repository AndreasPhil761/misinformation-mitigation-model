from datasets import load_dataset
import pandas as pd
import random
import pickle
from tqdm import tqdm
from extremity_scorer import score_emotional_extremity
from even_newer_extremity_scorer import scorer

#--------------------------------------------------------------------------
lt = 2000

lf = 2000
totl = lf + lt
x = 100
group_split = totl // x
#--------------------------------------------------------------------------
dataset = load_dataset("chengxuphd/liar2")

social_media_contexts = [

    "tweet", "tweets", "a Tweet", "Twitter",
    "facebook", "Facebook post", "FB", "facebook posts",
    "social media", "instagram", "Instagram",
    "online", "platform", "post", "status",
    "comment", "share", "viral", "thread",
    "campaign speech", "campaign speeches", "speech",
    "news release", "press release", "statement",
    "news", "article",

    "blog", "website", "web", "forum",
    "official statement","rally",
    "town hall", "conference"
]

social_media_true = dataset['train'].filter(
    lambda x: (x['label'] == 5 or x['label'] == 4) and x['context'] is not None and
              any(context.lower() in x['context'].lower() for context in social_media_contexts)
)

social_media_false = dataset['train'].filter(
    lambda x: (x['label'] == 1 or x['label'] == 0) and x['context'] is not None and
              any(context.lower() in x['context'].lower() for context in social_media_contexts)
)

social_media_true = social_media_true.select(range(min(lt, len(social_media_true))))
social_media_false = social_media_false.select(range(min(lf, len(social_media_false))))

true_claims_list = [{'content': item['statement'], 'label': 'true'} for item in social_media_true]
false_claims_list = [{'content': item['statement'], 'label': 'false'} for item in social_media_false]

result = true_claims_list + false_claims_list

print("Starting emotional extremity scoring...")
#result = score_emotional_extremity(result)
result = scorer(result)
print("Emotional extremity scoring complete!")
df = pd.DataFrame(result)
df_shuffled = df.sample(frac=1).reset_index(drop=True)

#df.to_csv('liar2_social_media_sample.csv', index=False)


def assign_tc(df_shuffled, group_split, timesteps):
    flattened_posts = []  # Changed from grouped_posts to flattened_posts
    for tc in range(timesteps):
        start_idx = tc * group_split
        end_idx = start_idx + group_split
        group = df_shuffled[start_idx:end_idx].to_dict(orient='records')
        for post in group:
            post['tc'] = tc
            flattened_posts.append(post)  # Append directly to main list
    return flattened_posts


grouped_posts = assign_tc(df_shuffled, group_split, x)

print(f"length false = {lt}, length true = {lf}, timesteps = {x}, group split = {group_split}")
print(f"total posts: {len(grouped_posts)}")
print("\nFirst 100 posts:")


with open('grouped_posts.pkl', 'wb') as f:
    pickle.dump(grouped_posts, f)