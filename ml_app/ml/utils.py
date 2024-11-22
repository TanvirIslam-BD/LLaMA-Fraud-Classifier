import os
import pickle

import pandas as pd
import numpy as np
from river.tree import HoeffdingTreeClassifier
from transformers import AutoTokenizer, AutoModel
from river.ensemble import LeveragingBaggingClassifier
from river.metrics import Accuracy
from sklearn.preprocessing import OneHotEncoder
import torch
from huggingface_hub import login
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
login("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")

# Load the model and tokenizer (make sure to use a smaller model if resources are limited)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

arf = LeveragingBaggingClassifier(
    model=HoeffdingTreeClassifier(),
    n_models=10,
    seed=42
)

metric = Accuracy()

def generate_text_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def preprocess_data(df):
    # Generate text embeddings
    text_features = (df["Customer"] + " " + df["Organisation"] + " " + df["Event"])
    text_embeddings = generate_text_embeddings(text_features.tolist())
    text_embeddings = np.array(text_embeddings)
    if text_embeddings.ndim == 1:
        text_embeddings = text_embeddings.reshape(-1, 1)  # Ensure 2D

    # Extract numerical features
    numerical_features = df[["Amount", "Tickets", "Service Charge", "Coupon amount"]].to_numpy()
    numerical_features = np.array(numerical_features)
    if numerical_features.ndim == 1:
        numerical_features = numerical_features.reshape(-1, 1)  # Ensure 2D

    # One-hot encode categorical features
    categorical_features = df[["Booking type", "Status", "Processor", "Currency"]]
    encoder = OneHotEncoder(sparse_output=False)
    categorical_encoded = encoder.fit_transform(categorical_features)
    categorical_encoded = np.array(categorical_encoded)
    if categorical_encoded.ndim == 1:
        categorical_encoded = categorical_encoded.reshape(-1, 1)  # Ensure 2D

    # Extract temporal features
    df["Date"] = pd.to_datetime(df["Date"])
    temporal_features = df["Date"].apply(
        lambda x: [x.year, x.month, x.day, x.dayofweek, x.hour, x.minute]
    ).tolist()
    temporal_features = np.array(temporal_features)
    if temporal_features.ndim == 1:
        temporal_features = temporal_features.reshape(-1, 1)  # Ensure 2D

    # Ensure all arrays have consistent row counts
    min_rows = min(
        text_embeddings.shape[0],
        numerical_features.shape[0],
        categorical_encoded.shape[0],
        temporal_features.shape[0]
    )

    # Trim arrays to match the smallest row count
    text_embeddings = text_embeddings[:min_rows]
    numerical_features = numerical_features[:min_rows]
    categorical_encoded = categorical_encoded[:min_rows]
    temporal_features = temporal_features[:min_rows]

    # Combine all features horizontally
    features = np.hstack((text_embeddings, numerical_features, categorical_encoded, temporal_features))

    # Extract target variable
    target = df["Genuine Order"].to_numpy()
    target = target[:min_rows]  # Ensure target matches the number of rows in features

    return df, features, target


def preprocess_data_for_prediction(df):
    # Text Features
    text_features = (df["Customer"] + " " + df["Organisation"] + " " + df["Event"])
    text_embeddings = generate_text_embeddings(text_features.tolist())
    text_embeddings = np.array(text_embeddings)
    if text_embeddings.ndim == 1:
        text_embeddings = text_embeddings.reshape(1, -1)

    # Numerical Features
    numerical_features = df[["Amount", "Tickets", "Service Charge", "Coupon amount"]].to_numpy()
    numerical_features = np.array(numerical_features)
    if numerical_features.ndim == 1:
        numerical_features = numerical_features.reshape(-1, 1)

    # Categorical Features
    categorical_features = df[["Booking type", "Status", "Processor", "Currency"]]
    encoder = OneHotEncoder(sparse_output=False)
    categorical_encoded = encoder.fit_transform(categorical_features)
    categorical_encoded = np.array(categorical_encoded)
    if categorical_encoded.ndim == 1:
        categorical_encoded = categorical_encoded.reshape(-1, 1)

    # Temporal Features
    df["Date"] = pd.to_datetime(df["Date"])
    temporal_features = df["Date"].apply(
        lambda x: [x.year, x.month, x.day, x.dayofweek, x.hour, x.minute]
    ).tolist()
    temporal_features = np.array(temporal_features)
    if temporal_features.ndim == 1:
        temporal_features = temporal_features.reshape(-1, 1)

    # Debugging Shapes
    print(f"Text Embeddings Shape: {text_embeddings.shape}")
    print(f"Numerical Features Shape: {numerical_features.shape}")
    print(f"Categorical Encoded Shape: {categorical_encoded.shape}")
    print(f"Temporal Features Shape: {temporal_features.shape}")

    # Ensure consistent row counts
    min_rows = min(
        text_embeddings.shape[0],
        numerical_features.shape[0],
        categorical_encoded.shape[0],
        temporal_features.shape[0]
    )

    text_embeddings = text_embeddings[:min_rows]
    numerical_features = numerical_features[:min_rows]
    categorical_encoded = categorical_encoded[:min_rows]
    temporal_features = temporal_features[:min_rows]

    # Combine features
    features = np.hstack((text_embeddings, numerical_features, categorical_encoded, temporal_features))
    return df, features


def initial_train_from_csv(csv_path, model, encoder_path="encoder.pkl", model_path="model.pkl"):
    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Preprocess data
    text_features = (df["Customer"] + " " + df["Organisation"] + " " + df["Event"])

    text_embeddings = generate_text_embeddings(text_features.tolist())

    numerical_features = df[["Amount", "Tickets", "Service Charge", "Coupon amount"]].to_numpy()

    # Encode categorical features
    categorical_features = df[["Booking type", "Status", "Processor", "Currency"]]
    encoder = OneHotEncoder(sparse=False)
    categorical_encoded = encoder.fit_transform(categorical_features)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)  # Save the encoder for future use

    # Process datetime features
    df["Date"] = pd.to_datetime(df["Date"])
    temporal_features = df["Date"].apply(
        lambda x: [x.year, x.month, x.day, x.dayofweek, x.hour, x.minute]
    ).tolist()

    # Combine features
    features = np.hstack((text_embeddings, numerical_features, categorical_encoded, temporal_features))
    target = df["Genuine Order"].to_numpy()

    # Initialize metric
    metric = Accuracy()

    # Train the model incrementally
    for i in range(len(features)):
        X_sample_dict = {f"feature_{j}": features[i][j] for j in range(len(features[i]))}
        y_sample = target[i]
        y_pred = model.predict_one(X_sample_dict)
        metric.update(y_sample, y_pred)
        model.learn_one(X_sample_dict, y_sample)

    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Initial training complete. Accuracy: {metric.get()}")
    return { Accuracy: metric.get()}
