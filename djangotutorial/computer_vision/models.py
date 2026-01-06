from django.db import models

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
