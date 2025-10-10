from .models import StoryboardHistory

def recent_histories(request):
    return {
        "recent_histories": StoryboardHistory.objects.all()[:10]
    }
