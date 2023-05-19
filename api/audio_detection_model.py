from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Item
from .serializers import ItemSerializer, VideoSerializer
import cv2
import numpy as np
import os
from django.conf import settings
from keras.models import load_model
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

        img = Image.open(output_path).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)

        print(f'Spectogram {img.shape}')

        return img

        # Print file name
        print(audio + ' ---------- Done')

    except Exception as e:
        print(audio + ' ---------- Failed')
    
def audio_detection_model(melspectogram):

    class_encoding = {0: 'fake', 1: 'real'}

    model = load_model('api/ResNet50_16_94.57.h5')
    predictions = model.predict(melspectogram)
    
    # Get predicted class label
    predicted_label_index = np.argmax(predictions)
    predicted_label = class_encoding[predicted_label_index]
    confidence_level = np.max(predictions) * 100

    # print(f"Audio Predicted class: {predicted_label} ({confidence_level:.2f}%)")
    
    return predicted_label, confidence_level
    