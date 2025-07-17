from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "cardiffnlp/twitter-roberta-large-emotion-latest"
emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
def measure_emotional_intensity(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    emotion_labels = ["anger", "anticipation", "disgust", "fear", "joy",
                     "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
    emotion_scores = {emotion: probabilities[i].item() for i, emotion in enumerate(emotion_labels)}
    target_emotions = ["anger", "fear", "disgust", "pessimism"]
    target_scores = sum(emotion_scores[emotion] for emotion in target_emotions)
    return min(1.0, target_scores)



'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load emotion classification model
model_name = "Panda0116/emotion-classification-model"
emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)

def measure_emotional_intensity(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = emotion_model(**inputs)

    # Get probabilities for each emotion class
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    # Map indices to emotion labels (match the model's categories)
    emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    emotion_scores = {emotion: probabilities[i].item() for i, emotion in enumerate(emotion_labels)}

    # Focus on target negative emotions
    target_emotions = ["anger", "fear"]
    target_scores = sum(emotion_scores[emotion] for emotion in target_emotions)

    # Return a single intensity score capped at 1.0
    return min(1.0, target_scores)
'''
'''
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()


# Function to transform compound score to 0-1 scale (0=neutral, 1=negative extreme)
def measure_emotional_intensity(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    # Transform: only negative compounds matter, and we flip the scale
    # so -1 becomes 1 (high negative extremity) and 0 stays 0 (neutral)
    if compound <= 0:
        return abs(compound)  # -0.8 becomes 0.8 (high negative extremity)
    else:
        return 0  # Positive sentiments are treated as neutral (0


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load emotion classification model
model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)


def measure_emotional_intensity(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = emotion_model(**inputs)

    # Get probabilities for each emotion class
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    # Map indices to emotion labels
    emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    emotion_scores = {emotion: probabilities[i].item() for i, emotion in enumerate(emotion_labels)}

    # Focus on target negative emotions
    target_emotions = ["anger", "fear", "disgust"]
    target_scores = sum(emotion_scores[emotion] for emotion in target_emotions)

    # Return a single intensity score capped at 1.0
    return min(1.0, target_scores)
'''

