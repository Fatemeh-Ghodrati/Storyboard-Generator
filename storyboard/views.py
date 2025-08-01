from django.shortcuts import render
from .forms import ScriptInputForm
from .utils import extract_visual_scenes, summarize_text, generate_image_from_prompt
import base64

def index(request):
    results = []
    if request.method == 'POST':
        form = ScriptInputForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['script_text']
            
            summarized_text = summarize_text(text)
            
            visual_scenes = extract_visual_scenes(summarized_text)

            for scene in visual_scenes:
                prompt = f"cinematic storyboard scene: {scene['text']} with keywords {', '.join(scene['keywords'])}"
                image_bytes = generate_image_from_prompt(prompt)
                if isinstance(image_bytes, bytes) and len(image_bytes) > 100:
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    results.append((scene['text'], image_base64, scene['keywords'], scene['sentiment'], scene['score']))
                else:
                    results.append((scene['text'], f"Error: {image_bytes.decode()}", scene['keywords'], scene['sentiment'], scene['score']))
    else:
        form = ScriptInputForm()

    return render(request, 'storyboard/index.html', {
        'form': form,
        'results': results
    })