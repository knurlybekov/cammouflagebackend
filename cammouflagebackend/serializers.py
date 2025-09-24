from rest_framework import serializers
from .models import UploadImageModel
from PIL import Image

class UploadImageSerializer(serializers.ModelSerializer):
    class Meta:
        model  = UploadImageModel
        fields = ["id", "image"]

    def validate_image(self, img):
        allowed = {"image/jpeg", "image/png", "image/webp"}
        ct = getattr(img, "content_type", None)
        if ct and ct not in allowed:
            raise serializers.ValidationError("Only JPEG/PNG/WebP are allowed.")
        if img.size > 5 * 1024 * 1024:
            raise serializers.ValidationError("Image too large (max 5 MB).")
        try:
            Image.open(img).verify()
        except Exception:
            raise serializers.ValidationError("Corrupted or unsupported image.")
        return img