import os
import ssl
import cv2
import io
import base64
import numpy as np
import tensorflow as tf
from django.shortcuts import render, redirect

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array

from .forms import ImageUploadForm, VideoUploadForm, AudioUploadForm
from .models import VideoAnalysis, AudioModel

import tensorflow_hub as hub
import pandas as pd
from scipy.io import wavfile
from django.conf import settings
from pydub import AudioSegment
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

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



# --- РОЗПІЗНАВАННЯ AУДІО ---
# -------------------------------------------------------------
MODEL_PATH = os.path.join(settings.BASE_DIR, 'computer_vision', 'models', 'yamnet')
# 2. Завантажуємо модель ЛОКАЛЬНО з папки (не з інтернету)
yamnet_model = hub.load(MODEL_PATH)

def audio_rec(request):
    result = None
    instance = None

    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        instance = AudioModel.objects.create(audio=audio_file)
        
        try:
            # 1. Читаємо файл незалежно від формату (MP3, WAV, M4A)
            audio = AudioSegment.from_file(instance.audio.path)
            
            # 2. Конвертуємо під вимоги моделі (16kHz, Mono)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # 3. Перетворюємо в масив чисел (масив амплітуд)
            samples = np.array(audio.get_array_of_samples())
            
            # 4. Нормалізація: переводимо цілі числа в float32 (від -1.0 до 1.0)
            waveform = samples.astype(np.float32) / 32768.0
            
            # 5. Прогноз нейромережі
            scores, embeddings, spectrogram = yamnet_model(waveform)
            
            # 6. Отримуємо назву класу з CSV
            class_map_path = os.path.join(MODEL_PATH, 'assets', 'yamnet_class_map.csv')
            class_names = pd.read_csv(class_map_path)['display_name'].tolist()
            
            top_class = np.argmax(np.mean(scores.numpy(), axis=0))
            result = class_names[top_class]

            instance.result = result
            instance.save()

        except Exception as e:
            result = f"Помилка обробки: {str(e)}"

    return render(request, 'computer_vision/audio_rec.html', {
        'result': result,
        'instance': instance
    })
# -------------------------------------------------------------



# --- СПЕКТРАЛЬНИЙ АНАЛІЗ ---
# -------------------------------------------------------------

def spectrum_rec(request):
    result = None
    instance = None
    spectrogram_img = None
    
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            
            try:
                # 1. Завантаження аудіо
                audio = AudioSegment.from_file(instance.audio.path)
                audio = audio.set_channels(1)  # Робимо моно
                
                # 2. Отримання числових даних
                samples = np.array(audio.get_array_of_samples())
                sample_rate = audio.frame_rate

                # 3. Побудова графіка
                plt.figure(figsize=(10, 4))
                plt.specgram(samples, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
                plt.title('Спектрограма аудіосигналу')
                plt.xlabel('Час (сек)')
                plt.ylabel('Частота (Гц)')
                plt.colorbar(label='Гучність (дБ)')
                plt.tight_layout()

                # 4. Перетворення графіка в картинку (Base64)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plt.close() # Очищаємо пам'ять

                # Заповнюємо змінну, яка раніше була порожньою
                spectrogram_img = base64.b64encode(image_png).decode('utf-8')
                
                result = "Спектральний аналіз успішно виконано"
            except Exception as e:
                result = f"Помилка при обробці: {e}"
        else:
            result = "Помилка завантаження: перевірте формат та розмір файлу."
    else:
        form = AudioUploadForm()

    return render(request, 'computer_vision/spectrum_rec.html', {
        'form': form,
        'result': result,
        'instance': instance,
        'spectrogram_img': spectrogram_img
    })
# -------------------------------------------------------------



def custom_rec(request):
    return render(request, 'computer_vision/custom_rec.html')