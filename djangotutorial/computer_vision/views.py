from django.shortcuts import render, redirect
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from django.shortcuts import render
from .forms import ImageUploadForm
import ssl
import os

# Вимикаємо перевірку SSL сертифікатів
ssl._create_default_https_context = ssl._create_unverified_context

def index(request):
    return redirect('computer_vision:image_rec')


# розпізнавання зображення 
# Завантажуємо модель один раз при запуску сервера
model = MobileNetV2(weights='imagenet')

def image_rec(request):
    result = None
    instance = None  # Важливо: ініціалізуємо змінну на початку
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            
            # 1. Підготовка зображення
            img_path = instance.image.path
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # 2. Розпізнавання
            preds = model.predict(x)
            
            # 3. Декодування результату
            decoded = decode_predictions(preds, top=1)[0][0]
            label = decoded[1].replace('_', ' ')
            score = decoded[2] * 100
            
            result = f"Це {label} з імовірністю {score:.2f}%"
            
            # Збереження результату в базу
            instance.analysis_result = {"label": label, "score": float(decoded[2])}
            instance.save()
    else:
        form = ImageUploadForm()

    # Передаємо instance, щоб картинка відобразилася в HTML
    return render(request, 'computer_vision/image_rec.html', {
        'form': form, 
        'result': result, 
        'instance': instance
    })


def video_rec(request):
    return render(request, 'computer_vision/video_rec.html')

def audio_rec(request):
    return render(request, 'computer_vision/audio_rec.html')

def spectrum_rec(request):
    return render(request, 'computer_vision/spectrum_rec.html')

def custom_rec(request):
    return render(request, 'computer_vision/custom_rec.html')