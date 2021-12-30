from django.db import models

# Create your models here.
class Uploadimage(models.Model):
    uploadimagefile = models.FileField(upload_to="")
    gender = models.CharField(max_length=15)
    skin = models.CharField(max_length=15)
    age = models.CharField(max_length=15)
    shape = models.CharField(max_length=15)
    forehead = models.CharField(max_length=15)
    eyebrows = models.CharField(max_length=15)
    eye_color = models.CharField(max_length=15)
    nose_length = models.CharField(max_length=15)
    nose_width = models.CharField(max_length=15)
    chin = models.CharField(max_length=15)
    mole = models.CharField(max_length=15)
    scar = models.CharField(max_length=15)
    moustaches = models.CharField(max_length=15)
    beard = models.CharField(max_length=15)
    filenames = models.CharField(max_length=30)
