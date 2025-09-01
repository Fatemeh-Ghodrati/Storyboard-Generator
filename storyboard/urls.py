from django.urls import path
from . import views

app_name = "storyboard"

urlpatterns = [
    path("", views.index, name="index"),
]
