from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Item
from .serializers import ItemSerializer, VideoSerializer
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.conf import settings
import moviepy.editor as mp
from .audio_detection_model import *
from .video_detection_model import *

@api_view(['GET'])
def getData(request):
    items = Item.objects.all()
    serializer = ItemSerializer(items, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def addItem(request):
    serializer = ItemSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)

@api_view(['POST'])
def addVideo(request):
    serializer = VideoSerializer(data=request.data)
    if serializer.is_valid():
        # print(request.data)
        video = serializer.validated_data['video']
        filename = os.path.join(video.name)
        path = default_storage.save(os.path.join(settings.MEDIA_ROOT, filename), ContentFile(video.read()))
        print(filename)

        # Read the video file using OpenCV
        video_path = os.path.join(settings.MEDIA_ROOT, path)
        audio_path = os.path.join(settings.MEDIA_ROOT, 'audio.wav')
    
        # capture = cv2.VideoCapture(video_path)

        video_classification, video_confidence = video_detection_model(video_path)
        print(f"c: {video_classification} ({video_confidence:.2f}%)")

        clip = mp.VideoFileClip(video_path)
    
        if clip.audio is not None:
            clip.audio.write_audiofile(audio_path)
            mel_spectogram = melspectogram(audio_path)
            audio_classification, audio_confidence = audio_detection_model(mel_spectogram)
        else:
            audio_classification = "No Audio"
            audio_confidence =  "No Audio"
    
        return Response({"video_classification": video_classification,
                         "video_confidence_level": video_confidence,
                         "audio_classification": audio_classification,
                         "audio_confidence_level": audio_confidence
                         })

    else:
        # Return an error response if the serializer is not valid
        return Response(serializer.errors, status=400)