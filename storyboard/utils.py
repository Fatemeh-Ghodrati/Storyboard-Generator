import nltk
from io import BytesIO
from PIL import Image
import requests

nltk.download('punkt')

HEADERS = {
    "Authorization": "Bearer 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"
}
GENERATE_IMAGE_API = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"
OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_generate(prompt: str, model: str = "gemma3:1b") -> str:
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_text(text: str, min_length: int = 50) -> str:
    prompt = f"Summarize the following scene in at least {min_length} words without asking for more input:\n\n{text}\n\nSummary:"
    return ollama_generate(prompt)

def analyze_sentiment(text: str) -> dict:
    prompt = f"""
Classify the sentiment of the following sentence as Positive, Negative, or Neutral.
Only return the label name.

Sentence:
{text}

Sentiment:
"""
    result = ollama_generate(prompt).lower()
    if "positive" in result:
        return {"label": "POSITIVE", "score": 0.9}
    elif "negative" in result:
        return {"label": "NEGATIVE", "score": 0.9}
    elif "neutral" in result:
        return {"label": "NEUTRAL", "score": 0.6}
    else:
        return {"label": "UNKNOWN", "score": 0.0}

def build_visual_prompt(scene_text: str) -> str:
    return f"""
You are a helpful assistant that analyzes movie scenes and identifies visually important elements for creating a storyboard.

Given a scene description (including narration or dialogue), extract the key visual elements that should be illustrated in a storyboard.
Focus only on physical settings, actions, emotions, lighting, characters’ appearance, or anything visual.

Do not include abstract ideas. Return only a comma-separated list of key visual elements.

Scene:
\"\"\"{scene_text}\"\"\"

Visual Elements:
"""

def extract_visual_keywords_with_gemma(scene_text: str, top_n: int = 5) -> list:
    prompt = build_visual_prompt(scene_text)
    result = ollama_generate(prompt)
    keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
    return keywords[:top_n]

def extract_visual_scenes(text: str) -> list:
    sentences = nltk.sent_tokenize(text)
    visual_scenes = []

    for sentence in sentences:
        sentiment = analyze_sentiment(sentence)
        sentiment_score = sentiment["score"] if sentiment["label"] != "UNKNOWN" else 0.0

        keywords = extract_visual_keywords_with_gemma(sentence)

        visual_keywords = ['scene', 'landscape', 'battle', 'sunset', 'fight', 'emotion', 'action', 'dark', 'light', 'city', 'forest']
        is_visual = any(k.lower() in visual_keywords for k in keywords)

        if sentiment_score > 0.7 or is_visual:
            visual_scenes.append({
                'text': sentence,
                'keywords': keywords,
                'sentiment': sentiment['label'],
                'score': sentiment_score
            })

    return visual_scenes

def generate_image_from_prompt(prompt: str) -> bytes:
    try:
        payload = {"inputs": prompt}
        response = requests.post(GENERATE_IMAGE_API, headers=HEADERS, json=payload, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if 'image' in content_type or 'octet-stream' in content_type:
            return response.content
        else:
            return f"Error: Unexpected content type: {content_type}".encode()

    except Exception as e:
        return f"Error in image generation: {str(e)}".encode()
