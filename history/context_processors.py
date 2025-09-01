from .models import StoryboardHistory

def recent_histories(request):
    return {
        "histories": StoryboardHistory.objects.all()[:10]
    }
