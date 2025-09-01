from django.shortcuts import render, get_object_or_404, redirect
from .models import StoryboardHistory

def history_list(request):
    """Show 10 last generated storyboards"""
    histories = StoryboardHistory.objects.all()[:10]
    return render(request, "history/history_list.html", {"histories": histories})

def history_detail(request, pk):
    """Show selected history detail"""
    history = get_object_or_404(StoryboardHistory, pk=pk)
    return render(request, "history/history_detail.html", {"history": history})

