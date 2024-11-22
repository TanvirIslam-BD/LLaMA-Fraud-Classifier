# ml_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),               # Accessible at /ml/
    path('retrain/', views.retrain, name='retrain'),   # Accessible at /ml/retrain/
    path('auto_tune/', views.auto_tune, name='auto_tune'),  # Accessible at /ml/auto_tune/
    path('data_set_info/', views.data_set_info, name='data_set_info'),  # Accessible at /ml/data_set_info/
    path('last_trained_features/', views.load_last_trained_features, name='load_last_trained_features'),  # Accessible at /ml/load_last_trained_features/
    path('training-history/', views.training_history, name='training_history'), # Accessible at /ml/training_history/
    path('history/', views.history, name='history'), # Accessible at /ml/history/
]
