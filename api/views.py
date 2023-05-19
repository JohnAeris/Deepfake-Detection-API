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
from tensorflow import keras
from keras.models import load_model
from .classifier import PositionalEmbedding, TransformerEncoder
import moviepy.editor as mp
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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
        audio_path = os.path.join(settings.MEDIA_ROOT, 'audio.wav')
    
        # capture = cv2.VideoCapture(video_path)

        video_classification, video_confidence = video_detection_model(video_path)
        print(f"c: {video_classification} ({video_confidence:.2f}%)")

        clip = mp.VideoFileClip(video_path)
    
        if clip.audio is not None:
            clip.audio.write_audiofile(audio_path)
            melspectogram(audio_path)
            audio_classification = "Real"
        else:
            audio_classification = "No Audio"
    
        return Response({"video_classification": video_classification,
                         "video_confidence_level": video_confidence,
                         "audio_classification": audio_classification})

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

def compute_optical_flow(frame_index_1, frame_index_2, i):
    try:
        frame1 = cv2.resize(frame_index_1, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame_index_2, (224, 224), interpolation=cv2.INTER_LINEAR)

        # print(f'Frame shape {frame_index_1.shape}')
        frame1 = frame1.astype(np.uint8)
        frame2 = frame2.astype(np.uint8)

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
        # cv2.imwrite(settings.MEDIA_ROOT +  str(i) + '.jpg', diff)
        
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
            flow = compute_optical_flow(frames[i], frames[i+1], i)
            flows.append(flow)
            # print(flow.shape)
        except IndexError:
            pass
    flows = np.array(flows)
    return np.array(flows)

def video_detection_model(video_path):

    extracted_frames = getFrame(video_path, 50)
    print('Numer of frames: {}'.format(extracted_frames.shape))

    optical_flow = compute_optical_flow_all(extracted_frames)
    print('Optical Flow Shape: {}'.format(optical_flow.shape))

    feature_extraction = load_model('api/resnet50-feature-extraction-network.h5')
    feature_extraction.compile(loss='binary_crossentropy', optimizer='adam')

    features = feature_extraction.predict(optical_flow)
    print('Features: {}'.format(features.shape))

    features_reshaped = features.reshape((-1, 48, 2048))
    print('Features reshaped: {}'.format(features_reshaped.shape))

    model = load_model('api/Celeb-DF-v2_resnet50_50_890-C10_EP100_SL49_ED2048_DD2048_NH8_86.24.h5', custom_objects={"PositionalEmbedding": PositionalEmbedding, "TransformerEncoder": TransformerEncoder})
    
    class_labels = ['fake', 'real']
    class_encoding = {0: 'fake', 1: 'real'}

    predictions = model.predict(features_reshaped)
    # print(f'Prediction Array: {predictions}')

    # Get predicted class label
    predicted_label_index = np.argmax(predictions)
    predicted_label = class_encoding[predicted_label_index]
    confidence_level = np.max(predictions) * 100

    pred_label = class_labels[np.argmax(predictions)]
    # print(f"Predicted class: {predicted_label} ({confidence_level:.2f}%)")

    return predicted_label, confidence_level

def melspectogram(audio):
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    duration = 2
    y, sr = librosa.load(audio, duration=duration)
    
    # Validate if file is corrupted or not
    try:
        if len(y) < duration:
            y = librosa.util.fix_length(y, sr*duration)

        # Compute mel-spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        # Pad mel-spectrogram if shorter than 2 seconds
        spec_len = log_spectrogram.shape[1]
        if spec_len < 2 * sr // hop_length:
            pad_width = 2 * sr // hop_length - spec_len
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (pad_width // 2, pad_width - pad_width // 2)), mode='constant', constant_values=-80)
            
        if len(y) == 0 or spec_len == 0:
            raise ValueError('File has no duration or is corrupted')
        
        # Plot mel-spectrogram
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.tight_layout()

        # Save plot as PNG image
        output_path = os.path.join(settings.MEDIA_ROOT, '/', os.path.splitext(audio)[0] + '_mel.png')
        plt.savefig(output_path, dpi=300)

        # Print file name
        print(audio + ' ---------- Done')

    except Exception as e:
        print(audio + ' ---------- Failed')