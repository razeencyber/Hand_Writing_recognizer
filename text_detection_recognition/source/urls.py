from django.urls import path 
from . import views


urlpatterns = [
    path('', views.home, name='app-home'),
    path('textDetect/', views.textDetect, name='app-textDetect'),
    path('textRecognize/', views.textRecognize, name='app-textRecognize')
    
]