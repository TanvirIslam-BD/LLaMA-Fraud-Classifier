import logging
import os
import re
import pandas as pd
from django.http import JsonResponse
import joblib
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from django_ml_app import settings
from ml_app.ml.autotune import model_train_with_auto_tune
from ml_app.ml.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, DATE_FEATURES, MODEL_PATH, PRICE_FEATURES
from ml_app.ml.models import TrainedFeatures, TrainingHistory
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc

from .predict import get_model
from .utils import preprocess_data, arf, metric, initial_train_from_csv, preprocess_data_for_prediction

logger = logging.getLogger(__name__)

MODEL_FILE_PATH = os.path.join(settings.BASE_DIR, 'ml_app', 'saved_models', 'model.pkl')
TRAINING_DATA_PATH = os.path.join(settings.BASE_DIR, 'ml_app', 'data', 'training_data.csv')


def train_model(data):

    data, feature_matrix_x, target_vector_y = load_and_prepare_data(data)

    for i in range(len(feature_matrix_x)):
        X_sample_dict = {f"feature_{j}": feature_matrix_x[i][j] for j in range(len(feature_matrix_x[i]))}
        y_sample = target_vector_y[i]
        y_pred = arf.predict_one(X_sample_dict)
        metric.update(y_sample, y_pred)
        arf.learn_one(X_sample_dict, y_sample)

    # Save the model after training
    joblib.dump(arf, 'ml_app/saved_models/model.pkl')

    # preload the model
    get_model()

    return {"accuracy": metric.get()}


def train_model_cron(json_data):

    print("------train_model_cron----------")
    print(json_data)
    df = pd.DataFrame(json_data)

    data, feature_matrix_x, target_vector_y = prepare_data_from_json(df)

    for i in range(len(feature_matrix_x)):
        X_sample_dict = {f"feature_{j}": feature_matrix_x[i][j] for j in range(len(feature_matrix_x[i]))}
        y_sample = target_vector_y[i]
        y_pred = arf.predict_one(X_sample_dict)
        metric.update(y_sample, y_pred)
        arf.learn_one(X_sample_dict, y_sample)

    # Save the model after training
    joblib.dump(arf, 'ml_app/saved_models/model.pkl')

    #preload the model
    get_model()

    print(f"accuracy: {metric.get()}")

    return {"accuracy": metric.get()}


def initial_train_model(data):
    csv_file = data['file_path']
    return initial_train_from_csv(csv_file, arf)




def save_training_data(performance_metrics, best_tune_params):
    # Save the data to the TrainingHistory model
    training_history = TrainingHistory.objects.create(
        learning_rate=best_tune_params["classifier__learning_rate"],
        max_iter=best_tune_params["classifier__max_iter"],
        max_leaf_nodes=best_tune_params["classifier__max_leaf_nodes"],
        min_samples_leaf=best_tune_params["classifier__min_samples_leaf"],
        accuracy=performance_metrics["accuracy"],
        precision=performance_metrics["precision"],
        recall=performance_metrics["recall"],
        f1_score=performance_metrics["f1_score"],
        roc_auc=performance_metrics.get("roc_auc"),  # Check if roc_auc is present
        confusion_matrix=performance_metrics["confusion"],
        classification_report=performance_metrics["report"],
        feature_importance=performance_metrics["feature_importance_data"]
    )
    return training_history  # Return the saved object for further use if needed


def calculate_model_performance(feature_matrix_x_test, y_test, model):

    target_vector_y_prediction = model.predict(feature_matrix_x_test)

    accuracy = accuracy_score(y_test, target_vector_y_prediction)
    precision = precision_score(y_test, target_vector_y_prediction, average='binary')
    recall = recall_score(y_test, target_vector_y_prediction, average='binary')
    f1 = f1_score(y_test, target_vector_y_prediction, average='binary')
    confusion = confusion_matrix(y_test, target_vector_y_prediction).tolist()
    report = classification_report(y_test, target_vector_y_prediction, output_dict=True)

    y_pred_proba = model.predict_proba(feature_matrix_x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    feature_importance_data = calculate_permutation_importance(model, feature_matrix_x_test, y_test)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion": confusion,
        "report": report,
        "roc_auc": roc_auc,
        "feature_importance_data": feature_importance_data
    }

def calculate_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    total_importance = sum(result.importances_mean)

    if total_importance > 0:
        return [
            (feature, (importance / total_importance) * 100)
            for feature, importance in zip(X_test.columns, result.importances_mean)
        ]
    else:
        return [(feature, 0) for feature in X_test.columns]

def load_and_prepare_data(data: dict):

    file_path = data['file_path']

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or invalid format.")

    data, feature_matrix_x, target_vector_y = preprocess_data(data)

    return  data, feature_matrix_x, target_vector_y

def prepare_data_from_json(data: dict):
    data, feature_matrix_x, target_vector_y = preprocess_data(data)
    return  data, feature_matrix_x, target_vector_y

def prepare_json_data_to_prediction(data: dict):
    data, feature_matrix_x = preprocess_data_for_prediction(data)
    return  data, feature_matrix_x


def clean_price_value(value):
    # Remove all characters except digits and the first decimal point
    cleaned_value = re.sub(r'[^\d.]', '', value.strip())

    # Handle cases with multiple decimal points by keeping only the first one
    parts = cleaned_value.split('.')
    if len(parts) > 2:
        cleaned_value = parts[0] + '.' + ''.join(parts[1:])

    # Convert to float
    try:
        return float(cleaned_value)
    except ValueError:
        return None  # or handle it as needed, e.g., return 0.0 or raise an exception

def clean_price_data(data):
    # Loop over each price feature in PRICE_FEATURES and apply the cleaning function
    for feature in PRICE_FEATURES:
        if feature in data.columns:
            data[feature] = data[feature].apply(
                lambda x: clean_price_value(x) if isinstance(x, str) else x
            )
    return data

def save_last_trained_features(categorical_features, numeric_features, date_features):
    # Save the features to the database
    trained_features = TrainedFeatures.objects.create(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        date_features=date_features
    )
    trained_features.save()  # Save instance to the database
    return trained_features

def load_last_trained_features():
    try:
        # Get the latest TrainedFeatures record
        last_trained_features = TrainedFeatures.objects.latest('timestamp')

        # Structure the data to be returned as JSON
        data = {
            "categorical_features": last_trained_features.categorical_features,
            "numeric_features": last_trained_features.numeric_features,
            "date_features": last_trained_features.date_features,
            "timestamp": last_trained_features.timestamp,
        }

        return JsonResponse(data)
    except TrainedFeatures.DoesNotExist:
        # If no trained features are found, return an error message
        return JsonResponse({"error": "No trained features found."}, status=404)


def extract_datetime_features(data_frame):
    for feature in DATE_FEATURES:
        if feature in data_frame.columns:
            # Convert the column to datetime format
            data_frame[feature] = pd.to_datetime(data_frame[feature], errors='coerce')

            # Extract datetime components
            data_frame[f"{feature}_year"] = data_frame[feature].dt.year
            data_frame[f"{feature}_month"] = data_frame[feature].dt.month
            data_frame[f"{feature}_day"] = data_frame[feature].dt.day
            data_frame[f"{feature}_hour"] = data_frame[feature].dt.hour
            data_frame[f"{feature}_minute"] = data_frame[feature].dt.minute

            # Drop the original date column
            data_frame = data_frame.drop(columns=[feature])

    return data_frame

# Load model if it exists
def load_model():
    if os.path.exists(MODEL_FILE_PATH):
        return joblib.load(MODEL_FILE_PATH)
    return None

# Save the model and metrics
def save_model(model):
    os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_FILE_PATH)

def initiate_model_training(feature_matrix_x, target_vector_y):

    # Split the data into training and testing sets
    feature_matrix_x_train, feature_matrix_x_test, target_vector_y_train, target_vector_y_test = train_test_split(
        feature_matrix_x, target_vector_y, test_size=0.2, random_state=42
    )

    try:
        # Train the model with tune
        best_model, best_tune_params, test_score = model_train_with_auto_tune(X=feature_matrix_x_train, y=target_vector_y_train, categorical_features=CATEGORICAL_FEATURES, numeric_features=NUMERIC_FEATURES)
    except ValueError as e:
        logger.error("Error during model fitting: %s", e)
        logger.debug("X_train columns: %s", feature_matrix_x_train.columns.tolist())
        logger.debug("Expected categorical features: %s", CATEGORICAL_FEATURES)
        logger.debug("Expected numeric features: %s", NUMERIC_FEATURES)
        raise

    print("----------")
    print("best tuning params:")
    print(best_tune_params)
    print("----------")
    # Save the trained model
    save_model(best_model)

    # Return the model and test data for evaluation
    return best_model, target_vector_y_test, feature_matrix_x_test, best_tune_params