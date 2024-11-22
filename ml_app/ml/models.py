# ml_app/models.py
from django.db import models

class TrainedFeatures(models.Model):
    categorical_features = models.JSONField()
    numeric_features = models.JSONField()
    date_features = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)  # Optional: Track when the features were saved

    def __str__(self):
        return f"TrainedFeatures at {self.timestamp}"



class TrainingHistory(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)

    # Hyperparameters
    learning_rate = models.FloatField()
    max_iter = models.IntegerField()
    max_leaf_nodes = models.IntegerField()
    min_samples_leaf = models.IntegerField()

    # Performance metrics
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField(null=True, blank=True)
    confusion_matrix = models.JSONField()
    classification_report = models.JSONField()
    feature_importance = models.JSONField()

    def __str__(self):
        return f"Training History at {self.timestamp}"