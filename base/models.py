from django.db import models

# Create your models here.
class Item(models.Model):
    name = models.CharField(max_length=200)
    created = models.DateTimeField(auto_now_add=True)

# Create your models here.
class Video(models.Model):
    video = models.FileField()
    created = models.DateTimeField(auto_now_add=True)