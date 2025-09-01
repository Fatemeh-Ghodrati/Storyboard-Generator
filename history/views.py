from django.shortcuts import render, get_object_or_404, redirect
from .models import StoryboardHistory

def history_list(request):
    """نمایش 10 مورد اخیر در سایدبار"""
    histories = StoryboardHistory.objects.all()[:10]
    return render(request, "history/history_list.html", {"histories": histories})

def history_detail(request, pk):
    """نمایش جزئیات یک تاریخچه"""
    history = get_object_or_404(StoryboardHistory, pk=pk)
    return render(request, "history/history_detail.html", {"history": history})

