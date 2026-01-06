from django.shortcuts import render

def index(request):
    return render(request, 'computer_vision/base_dashboard.html')

def image_rec(request):
    return render(request, 'computer_vision/image_rec.html')

def video_rec(request):
    return render(request, 'computer_vision/video_rec.html')

def audio_rec(request):
    return render(request, 'computer_vision/audio_rec.html')

def spectrum_rec(request):
    return render(request, 'computer_vision/spectrum_rec.html')

def custom_rec(request):
    return render(request, 'computer_vision/custom_rec.html')