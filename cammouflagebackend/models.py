from django.conf import settings
from django.db import models
import uuid

class UploadImageModel(models.Model):
    image = models.ImageField(upload_to='images')
    def __str__(self):
        return self.caption