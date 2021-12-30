from django.urls import path
from . import views

app_name = "faceapp"

urlpatterns = [
    path('', views.index, name='index'),
    path('result', views.result, name='result'),
    path('error', views.error, name='error'),
    path('automatic_tagging', views.automatic_tagging, name='automatic_tagging'),
    path('suspect_retrieval', views.suspect_retrieval, name='suspect_retrieval'),
    path('suspect_retrieval_result', views.suspect_retrieval_result, name='suspect_retrieval_result'),
    path('tryanotherone', views.tryanotherone, name='tryanotherone')
]
