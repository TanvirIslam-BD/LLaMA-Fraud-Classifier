from celery import shared_task
from .ml import retrain, tuner

@shared_task
def scheduled_retrain():
    retrain.retrain_model()

@shared_task
def scheduled_tune():
    tuner.tune_model()