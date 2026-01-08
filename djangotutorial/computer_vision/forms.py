from django import forms
from .models import ImageAnalysis, VideoAnalysis


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageAnalysis
        fields = ['image']
        labels = {
            'image': 'Оберіть зображення для розпізнавання',
        }


class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoAnalysis
        fields = ['video']
        widgets = {
            'video': forms.FileInput(attrs={'accept': 'video/*'}),
        }