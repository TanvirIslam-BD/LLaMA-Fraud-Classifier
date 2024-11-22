# my_app/api/urls.py

from django.urls import path
from .views import PredictAPIView

urlpatterns = [
    path('predict/', PredictAPIView.as_view(), name='predict_api'),
]
