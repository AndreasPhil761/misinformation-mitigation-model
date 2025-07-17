# Install required libraries first:
# pip install textblob vaderSentiment nltk

import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
# Download required NLTK resources upfront



def analyze_text(text):
    """
    Returns a combined score (0-1) where:
    0 = Objective + Neutral/Positive
    1 = Subjective + Negative
    """
    # Subjectivity analysis (TextBlob + MPQA)
    tb_subjectivity = TextBlob(text).sentiment.subjectivity  # 0-1 scale

    # MPQA lexicon matches
    mpqa_words = set(opinion_lexicon.words())
    tokens = word_tokenize(text.lower())
    mpqa_score = len([w for w in tokens if w in mpqa_words]) / len(tokens) if tokens else 0

    combined_subjectivity = (tb_subjectivity + mpqa_score) / 2
    # Sentiment analysis (VADER)
    vader = SentimentIntensityAnalyzer()
    sentiment = vader.polarity_scores(text)['compound']  # [-1, +1]

    # Convert sentiment to 0-1 scale (0=neutral/positive, 1=negative)
    sentiment_score = (1 - (sentiment + 1) / 2)

    # Final combined score
    return round((combined_subjectivity + sentiment_score) / 2, 4)

'''
# Example usage
sample_texts = [
    "The chemical formula for water is H₂O",  # Objective + neutral
    "I absolutely despise rainy Mondays",  # Subjective + negative
    "Many experts suggest moderate exercise",  # Mildly subjective + neutral
    "This is the worst possible outcome!"  # Subjective + negative
]
'''
def scorer(texts):
    for text in texts:
        content = text['content']
        extremity_score  = analyze_text(content)
        print(extremity_score)
        text['extremity'] = extremity_score
    return texts
        #print(f"Text: {text}\nScore: {score}\n{'━' * 40}")