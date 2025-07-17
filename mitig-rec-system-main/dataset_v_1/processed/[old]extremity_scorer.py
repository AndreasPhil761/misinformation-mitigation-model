from emotional_scoring import measure_emotional_intensity
from political_scoring import classify_political_ideology

def score_emotional_extremity(posts):
    for post in posts:
        content = post['content']

        # Get political direction (-1 to 1, Republican to Democrat)
        direction = classify_political_ideology(content)

        # Get emotional intensity (0 to 1)
        intensity = measure_emotional_intensity(content)

        # Calculate extremity score
        extremity_score = direction * intensity

        post['extremity'] = extremity_score
        post['political_leaning'] = "Democrat" if direction > 0 else "Republican"
        post['emotional_intensity'] = intensity

    return posts


