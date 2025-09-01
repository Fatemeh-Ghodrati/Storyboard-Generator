from django.db import models

from django.db import models
from django.utils import timezone

class StoryboardHistory(models.Model):
    title = models.CharField(max_length=255)
    input_text = models.TextField()
    results = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title
