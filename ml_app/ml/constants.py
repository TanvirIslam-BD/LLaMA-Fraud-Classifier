# ml_app/constants.py

import os
from typing import List
from django.contrib import admin

# Customize the admin title
admin.site.site_header = "My Custom Admin Panel"
admin.site.site_title = "My Admin Portal"
admin.site.index_title = "Welcome to the Custom Admin Dashboard"


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TIMESTAMP_FMT = "%m-%d-%Y, %H:%M:%S"

MODEL_PATH = "data/pipeline.pkl"  # Path to save/load model
METRICS_PATH = "data/metrics.json"  # Path to save metrics
TRAIN_HISTORY_PATH = "data/train_history.json"
features_file = 'data/last_trained_features.json'

last_trained_features = []

LABEL: str = "Genuine Order"


NUMERIC_FEATURES: List[str] = [
    "Tickets",
]

PRICE_FEATURES: List[str] = [
    "Amount",
    "Service Charge",
    "Coupon amount"
]

CATEGORICAL_FEATURES: List[str] = [
    "Customer",
    "Organisation",
    "Event",
    "Processor",
    "Booking type",
    "Refund status",
    "Status",
    "Currency"
]


DATE_FEATURES: List[str] = [
    "Date",
]

DATE_EXTRACT_FEATURES: List[str] = [
    "year",
    "month",
    "day",
    "hour",
    "minute"
]

HYPERPARAMETERS  = {
    "learning_rate": 0.05,        # Moderately low learning rate for gradual learning
    "max_iter": 250,              # Sufficient iterations for convergence at this learning rate
    "max_leaf_nodes": 20,         # Balanced tree complexity
    "min_samples_leaf": 15        # Ensures each leaf has enough samples for generalization
}

# General constants
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB in bytes
SUPPORTED_FILE_TYPES = ['csv', 'xlsx']

# Model-related constants
DEFAULT_LEARNING_RATE = 0.05
MAX_ITERATIONS = 200
MAX_LEAF_NODES = 20
MIN_SAMPLES_LEAF = 15

# Message constants
ERROR_MESSAGES = {
    'file_too_large': 'The uploaded file is too large.',
    'unsupported_file_type': 'Unsupported file type.',
    'missing_column': 'Required column is missing from the dataset.',
}
