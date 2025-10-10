import nltk
import base64
import requests
import logging

nltk.download('punkt')

logger = logging.getLogger('storyboard')

HEADERS = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

GENERATE_IMAGE_API = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"
TEXT_GENERATION_API = "https://router.huggingface.co/v1/chat/completions"

IMAGE_REQUEST_COUNT = 0
MAX_IMAGE_REQUESTS = 5


def generate_text(prompt: str) -> str:
    try:
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_new_tokens": 300
        }
        response = requests.post(TEXT_GENERATION_API, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return f"Error: {str(e)}"


def summarize_text(text: str, min_length: int = 50) -> str:
    prompt = f"Summarize the following scene in at least {min_length} words:\n\n{text}\n\nSummary:"
    result = generate_text(prompt)
    logger.debug(f"summarize result received: {result}")
    return result


def analyze_sentiment(text: str) -> dict:
    prompt = f"""
Classify the sentiment of the following sentence as Positive, Negative, or Neutral.
Only return the label name.

Sentence:
{text}

Sentiment:
"""
    result = generate_text(prompt).lower()
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


def extract_visual_keywords(scene_text: str, top_n: int = 5) -> list:
    prompt = build_visual_prompt(scene_text)
    result = generate_text(prompt)
    keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
    logger.debug(f"extract visual kw result received: {keywords[:top_n]}")
    return keywords[:top_n]


def group_visual_units(text: str) -> list:
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
    response = generate_text(prompt)
    units = []
    for line in response.split("\n"):
        if line.strip().startswith(tuple(str(i) + '.' for i in range(1, 100))):
            parts = line.split('.', 1)
            if len(parts) == 2:
                units.append(parts[1].strip())
    logger.debug(f"summarize result received: {units}")
    return units


def generate_image_from_prompt(prompt: str) -> str:
    global IMAGE_REQUEST_COUNT

    if IMAGE_REQUEST_COUNT >= MAX_IMAGE_REQUESTS:
        logger.warning("Image generation limit reached (max 5). No more requests will be sent.")
        return None

    try:
        payload = {"inputs": prompt}
        response = requests.post(GENERATE_IMAGE_API, headers=HEADERS, json=payload)
        response.raise_for_status()

        IMAGE_REQUEST_COUNT += 1 

        content_type = response.headers.get('content-type', '').lower()

        if 'application/json' in content_type:
            data = response.json()
            if isinstance(data, dict) and 'generated_image' in data:
                image_base64 = data['generated_image']
            elif isinstance(data, list) and 'generated_image' in data[0]:
                image_base64 = data[0]['generated_image']
            else:
                logger.warning(f"JSON received but no image found: {data}")
                return None
            logger.debug("Image received from JSON response")
            return f"data:image/png;base64,{image_base64}"

        elif 'image' in content_type or 'octet-stream' in content_type:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            logger.debug("Image received as raw content")
            return f"data:image/png;base64,{image_base64}"

        else:
            logger.warning(f"Unexpected content type: {content_type}")
            return None

    except Exception as e:
        logger.error(f"Error in image generation: {e}")
        return None
