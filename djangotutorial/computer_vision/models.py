from django.db import models
from django.dispatch import receiver
import os

class ImageAnalysis(models.Model):
    # Поле для самого зображення
    image = models.ImageField(upload_to='recognition/images/', verbose_name="Зображення")
    
    # Дата завантаження
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата завантаження")
    
    # Поле для результату (зберігатимемо JSON від TensorFlow)
    analysis_result = models.JSONField(null=True, blank=True, verbose_name="Результат аналізу") 
    
    class Meta:
        verbose_name = "Аналіз зображення"
        verbose_name_plural = "Аналізи зображень"

    def __str__(self):
        return f"Аналіз №{self.id} від {self.uploaded_at.strftime('%d.%m.%Y %H:%M')}"


class VideoAnalysis(models.Model):
    video = models.FileField(upload_to='videos/', verbose_name="Відеофайл")
    analysis_result = models.JSONField(null=True, blank=True)  
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Video {self.id} analyzed at {self.uploaded_at}"
    

class AudioModel(models.Model):
    # Файли будуть зберігатися в папці media/audio/
    audio = models.FileField(upload_to='audio/', verbose_name="Аудіофайл")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # Поле для збереження результату розпізнавання (опціонально)
    result = models.CharField(max_length=255, blank=True, null=True, verbose_name="Результат аналізу")

    def __str__(self):
        return f"Запис {self.id} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        verbose_name = "Аудіозапис"
        verbose_name_plural = "Аудіозаписи"

# Метод для автоматичного видалення файлу з папки при видаленні об'єкта з БД
@receiver(models.signals.post_delete, sender=AudioModel)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    if instance.audio:
        if os.path.isfile(instance.audio.path):
            os.remove(instance.audio.path)