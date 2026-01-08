import os
import ssl
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render, redirect

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array

from .forms import ImageUploadForm, VideoUploadForm
from .models import VideoAnalysis

ssl._create_default_https_context = ssl._create_unverified_context

# Використовуємо одну модель і для фото, і для відео
model = MobileNetV2(weights='imagenet')

def index(request):
    return redirect('computer_vision:image_rec')



# --- РОЗПІЗНАВАННЯ ЗОБРАЖЕННЯ ---
# -------------------------------------------------------------
def image_rec(request):
    result = None
    instance = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            img_path = instance.image.path
            
            img = load_img(img_path, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            decoded = decode_predictions(preds, top=1)[0][0]
            label = decoded[1].replace('_', ' ')
            score = decoded[2] * 100
            
            result = f"Це {label} з імовірністю {score:.2f}%"
            instance.analysis_result = {"label": label, "score": float(decoded[2])}
            instance.save()
    else:
        form = ImageUploadForm()

    return render(request, 'computer_vision/image_rec.html', {
        'form': form, 'result': result, 'instance': instance
    })
# -------------------------------------------------------------



# --- РОЗПІЗНАВАННЯ ВІДЕО ---
# -------------------------------------------------------------
def video_rec(request):
    result = None
    instance = None
    
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            video_path = instance.video.path
            
            cap = cv2.VideoCapture(video_path)
            detected_objects = set()  # Зберігаємо унікальні назви знайдених об'єктів
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Аналізуємо кожен 30-й кадр (приблизно 1 кадр на секунду відео)
                if frame_idx % 30 == 0:
                    # Перетворюємо кадр OpenCV (BGR) у формат для моделі (RGB + Size)
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    
                    x = img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    
                    preds = model.predict(x, verbose=0)
                    decoded = decode_predictions(preds, top=3)[0] # Беремо топ-3 для кращого результату
                    
                    for _, label, score in decoded:
                        if score > 0.25:  # Якщо модель впевнена на 25%+
                            detected_objects.add(label.replace('_', ' '))
                
                frame_idx += 1
                # Обмеження для безпеки (наприклад, не більше 300 кадрів / 10 сек)
                if frame_idx > 300: 
                    break
            
            cap.release()
            
            if detected_objects:
                result = "На відео помічено: " + ", ".join(list(detected_objects))
            else:
                result = "Об'єктів не розпізнано."
                
            instance.analysis_result = {"detected": list(detected_objects)}
            instance.save()
    else:
        form = VideoUploadForm()

    return render(request, 'computer_vision/video_rec.html', {
        'form': form, 'result': result, 'instance': instance
    })
# -------------------------------------------------------------



# --- РЕШТА ФУНКЦІЙ ---
def audio_rec(request):
    return render(request, 'computer_vision/audio_rec.html')

def spectrum_rec(request):
    return render(request, 'computer_vision/spectrum_rec.html')

def custom_rec(request):
    return render(request, 'computer_vision/custom_rec.html')