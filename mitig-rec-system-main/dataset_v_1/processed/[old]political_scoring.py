import random


def classify_political_ideology(text):

    political_wing = random.choice([1, -1])

    return political_wing


'''
from transformers import pipeline

# Load Political DEBATE model for zero-shot political classification
political_classifier = pipeline("zero-shot-classification",
                                model="mlburnham/Political_DEBATE_large_v1.0")


def classify_political_ideology(text):
    result = political_classifier(
        text,
        ["this was written by a Democrat", "this was written by a Republican"]
    )

    # Determine political direction based on scores
    if result['labels'][0] == "this was written by a Republican":
        direction = -1 * result['scores'][0]  # Negative for Republican
    else:
        direction = result['scores'][0]  # Positive for Democrat

    return direction
'''