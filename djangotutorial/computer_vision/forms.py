from django import forms
from .models import ImageAnalysis, VideoAnalysis, AudioModel


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


class AudioUploadForm(forms.ModelForm):
    class Meta:
        model = AudioModel
        fields = ['audio']
        widgets = {
            'audio': forms.FileInput(attrs={'accept': 'audio/*', 'class': 'form-control-file'})
        }

    # Додамо валідацію розміру файлу (наприклад, до 10 МБ)
    def clean_audio(self):
        audio = self.cleaned_data.get('audio')
        if audio:
            if audio.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Файл занадто великий (макс. 10 МБ)")
        return audio