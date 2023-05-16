from rest_framework import serializers
from base.models import Item, Video

class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Item
        fields = '__all__' 

class VideoSerializer(serializers.ModelSerializer):
    video = serializers.FileField()

    class Meta:
        model = Video
        fields = '__all__' 