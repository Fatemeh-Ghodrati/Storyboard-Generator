from django.shortcuts import render
from .forms import ScriptInputForm
from .utils import (
    summarize_text,
    group_visual_units,
    extract_visual_keywords_with_gemma,
    generate_image_from_prompt,
    analyze_sentiment
)
import base64

def index(request):
    results = []
    if request.method == 'POST':
        form = ScriptInputForm(request.POST)
        if form.is_valid():
            script = form.cleaned_data['script_text']

            summarized = summarize_text(script)
            visual_units = group_visual_units(summarized)

            for unit in visual_units:
                keywords = extract_visual_keywords_with_gemma(unit)
                sentiment = analyze_sentiment(unit)
                prompt = f"cinematic storyboard scene: {unit} with keywords {', '.join(keywords)}"
                image_url = generate_image_from_prompt(prompt)

                results.append((unit, image_url, keywords, sentiment['label'], sentiment['score']))
    else:
        form = ScriptInputForm()

    return render(request, 'storyboard/index.html', {
        'form': form,
        'results': results
    })

