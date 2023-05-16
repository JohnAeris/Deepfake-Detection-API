from rest_framework import serializers
from base.models import Item

class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Item
        fields = '__all__' 

class VideoSerializer(serializers.Serializer):
    video = serializers.FileField()