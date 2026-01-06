from django.urls import path
from . import views

app_name = 'computer_vision'

urlpatterns = [
    path('', views.index, name='index'),
    path('image/', views.image_rec, name='image_rec'),
    path('video/', views.video_rec, name='video_rec'),
    path('audio/', views.audio_rec, name='audio_rec'),
    path('spectrum/', views.spectrum_rec, name='spectrum_rec'),
    path('custom/', views.custom_rec, name='custom_rec'),
]