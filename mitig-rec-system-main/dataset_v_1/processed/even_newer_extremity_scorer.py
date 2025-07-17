from mistralai import Mistral
import json
import time

api_key = #INSERT YOUR OWN API KEY HERE 
client = Mistral(api_key=api_key)


def analyze_texts_batch(texts_batch, batch_size=5):
    """Analyze multiple texts in a single API call with improved prompt"""
    # Create batch prompt with numbered texts
    texts_content = "\n".join([f"Text {i + 1}: \"{text['content']}\"" for i, text in enumerate(texts_batch)])

    prompt = f"""Analyze each text for how it attempts to evoke emotional extremity for a reader. Return only JSON array with analysis for each text:
    [
      {{
        "text_id": 1,
        "emotions": {{
          "anger": 0-1 (0=not trying to evoke anger, 1=strongly trying to evoke anger),
          "anxiety": 0-1 (0=not trying to evoke anxiety, 1=strongly trying to evoke anxiety),
          "shock": 0-1 (0=not trying to evoke shock, 1=strongly trying to evoke shock),
          "fear": 0-1 (0=not trying to evoke fear, 1=strongly trying to evoke fear)
        }},
        "subjectivity": 0-1 (0=very objective language, 1=very subjective language),
        "sentiment": 0-1 (0=neutral or positive sentiment, 1=very negative sentiment)
      }},
      {{
        "text_id": 2,
        ...
      }},
      ...
    ]

    {texts_content}
    """

    # Make a single API call for the batch
    response = client.chat.complete(
        model="open-mistral-nemo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300 * len(texts_batch),  # Scale max tokens based on batch size
        response_format={"type": "json_object"}
    )

    try:
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Batch analysis error: {e}")
        print(f"Response: {response.choices[0].message.content}")
        return []


def calculate_extremity_score(analysis):
    """Calculate extremity score from analysis result"""
    emotions = analysis.get('emotions', {})
    if not emotions:
        return 0.0

    emotion_avg = sum(emotions.values()) / len(emotions)
    return round(
        emotion_avg * 0.6 +
        analysis.get('subjectivity', 0) * 0.2 +
        analysis.get('sentiment', 0) * 0.2,
        4
    )


def scorer(texts, batch_size=10):
    """Process texts in batches"""
    results = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Get analysis for the current batch
        analyses = analyze_texts_batch(batch, batch_size)

        # Match analyses with original texts and calculate scores
        for analysis in analyses:
            if 'text_id' in analysis:
                batch_index = analysis['text_id'] - 1
                if 0 <= batch_index < len(batch):
                    score = calculate_extremity_score(analysis)
                    batch[batch_index]['extremity'] = score

                    # Print results
                    print(f"Text: {batch[batch_index]['content']}")
                    print(f"Label: {batch[batch_index]['label']}")
                    print(f"Extremity Score: {score}")

        # Add delay between batches, not between individual texts
        if i + batch_size < len(texts):
            time.sleep(2)  # Wait between batches

    return texts


