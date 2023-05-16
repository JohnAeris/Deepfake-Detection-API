from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Item
from .serializers import ItemSerializer, VideoSerializer
import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.conf import settings
from facenet_pytorch import MTCNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(select_largest=False, post_process=False, device=device)

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
        print(video_path)
        # print('test')
        # capture = cv2.VideoCapture(video_path)

        extracted_frames = getFrame(video_path, 50)
        print('Numer of frames: {}'.format(extracted_frames.shape))

        optical_flow = compute_optical_flow_all(extracted_frames)
        print('Optical Flow Shape: {}'.format(optical_flow.shape))
        
        return Response({"message": "Video is being processed"})

    else:
        # Return an error response if the serializer is not valid
        return Response(serializer.errors, status=400)

def getFrame(video, num_frame):
    # video_path = os.path.join(source_dir, video)
    vidcap = cv2.VideoCapture(video)
    # print(vidcap)

    frames = []

    def saveFrame(count):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, count-1)
        # print(vidcap.set(cv2.CAP_PROP_POS_FRAMES, count-1))
        hasFrames, image = vidcap.read()
        if hasFrames:
            try:
                # process(target_dir, video, image, count)
                frame_read = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detected_face = detector(frame_read)
                face_np = detected_face.numpy()
                face_np = np.transpose(face_np, (1, 2, 0))
                face_cv2 = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                face_np = cv2.resize(face_cv2, (224, 224), interpolation=cv2.INTER_LINEAR)
                # print('Face shape {}'.format(face_np.shape))
                # cv2.imwrite('D:/Deepfake-Detection-Api/media/' + str(count) + '.jpg', face_np)
                frames.append(face_np)
            except AttributeError:
                pass
        return hasFrames

    for count in range(1, num_frame+1):
        # print(count)
        success = saveFrame(count)
        if not success:
            break
    
    npframes = np.array(frames)

    return npframes

def compute_optical_flow(frames, frame_index_1, frame_index_2, i):
    # Convert numpy array to list of frames
    frame_list = [np.array(frames[i]) for i in range(50)]
    # Convert frame indices to zero-based index
    try:
        frame1 = cv2.resize(frame_index_1, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame_index_2, (224, 224), interpolation=cv2.INTER_LINEAR)

        print(frame_index_1.shape)
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute the optical flow between the frames
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert flow vectors to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold the magnitude to create a binary mask of moving regions
        magnitude_threshold = 1
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ret, mask = cv2.threshold(magnitude, magnitude_threshold, 1, cv2.THRESH_BINARY)

        # Perform erosion to remove small noise or artifacts
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)

        # Perform dilation to fill gaps in the motion mask
        mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=1)

        # Apply the mask to the second frame to highlight the moving regions
        frame2_masked = cv2.bitwise_and(frame2, frame2, mask=mask_dilated)

        diff = cv2.absdiff(frame1, frame2_masked)
        diff = cv2.resize(diff, (224, 224), interpolation=cv2.INTER_LINEAR)
        diff = cv2.applyColorMap(diff * 255, cv2.COLORMAP_JET)
        cv2.imwrite('D:/Deepfake-Detection-Api/media/' +  str(i) + '.jpg', diff)
        
        return diff
    
    except IndexError:
        pass

def compute_optical_flow_all(frames):
    flows = []
    num_frames = frames.shape[0]
    # print(num_frames)
    for i in range(1, num_frames):
        # print(frames[i].shape)
        # print(i)
        try:
            flow = compute_optical_flow(frames, frames[i], frames[i+1], i)
            flows.append(flow)
            # print(flow.shape)
        except IndexError:
            pass
    flows = np.array(flows)
    return np.array(flows)