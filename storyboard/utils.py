import nltk
import base64
import requests
import logging

nltk.download('punkt')

logger = logging.getLogger('storyboard')

HEADERS = {
    "Authorization": "Bearer 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"
}
GENERATE_IMAGE_API = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"
OLLAMA_URL = "http://localhost:11434/api/generate"

DEFAULT_MODEL = "mistral:7b-instruct-q4_0" 

def ollama_generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
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

def summarize_text(text: str, min_length: int = 50, model: str = DEFAULT_MODEL) -> str:
    prompt = f"Summarize the following scene in at least {min_length} words:\n\n{text}\n\nSummary:"
    result = ollama_generate(prompt, model)
    logger.debug(f"summarize result received: {result}")
    return result

def analyze_sentiment(text: str, model: str = DEFAULT_MODEL) -> dict:
    prompt = f"""
Classify the sentiment of the following sentence as Positive, Negative, or Neutral.
Only return the label name.

Sentence:
{text}

Sentiment:
"""
    result = ollama_generate(prompt, model).lower()
    logger.debug(f"analyze result received: {result}")
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
You are a storyboard expert that identifies visually important elements for creating a storyboard.

Given a scene description (including narration or dialogue), extract the key visual elements that should be illustrated in a storyboard.
Focus ONLY on physical settings, actions, emotions, lighting, characters’ appearance, or anything visual.
Do not include abstract ideas. 
Return only a comma-separated list of key visual elements.

Scene:
\"\"\"{scene_text}\"\"\"

Visual Elements:
"""

def extract_visual_keywords_with_gemma(scene_text: str, top_n: int = 5, model: str = DEFAULT_MODEL) -> list:
    prompt = build_visual_prompt(scene_text)
    result = ollama_generate(prompt, model)
    keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
    logger.debug(f"extract visual kw result received: {keywords[:top_n]}")
    return keywords[:top_n]

def group_visual_units(text: str, model: str = DEFAULT_MODEL) -> list:
    prompt = f"""
You are a storyboard expert helping to split a story or script into the FEWEST possible distinct visual units.

Only include moments where a significant change happens in setting, action, or visual composition.
If multiple sentences describe the same visual moment, merge them into one unit.
Avoid over-splitting. 
Return ONLY a numbered list of the important visual moments without extra explanation.

Text:
\"\"\"{text}\"\"\"

Visual Units:
"""
    response = ollama_generate(prompt, model)
    units = []
    for line in response.split("\n"):
        if line.strip().startswith(tuple(str(i) + '.' for i in range(1, 100))):
            parts = line.split('.', 1)
            if len(parts) == 2:
                units.append(parts[1].strip())
    logger.debug(f"summarize result received: {units}")
    return units

def generate_image_from_prompt(prompt: str) -> str:
    try:
        payload = {"inputs": prompt}
        response = requests.post(GENERATE_IMAGE_API, headers=HEADERS, json=payload, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if 'image' in content_type or 'octet-stream' in content_type:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        else:
            return None

    except Exception as e:
        return None
