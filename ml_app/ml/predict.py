import joblib
from numpy.random import Generator, PCG64
from ml_app.ml.train import prepare_json_data_to_prediction
from ml_app.ml.utils import get_model

rng = Generator(PCG64())
print(rng.random())


def prediction(data):
    # try:

        # Load the saved model
        model = get_model()

        # Preprocessing the json data
        data, feature_matrix_x = prepare_json_data_to_prediction(data)

        # Check if it's a River model or a Pipeline
        if hasattr(model, 'predict_one'):
            # Transform feature matrix into a dictionary for River models
            feature_dict = {f"feature_{i}": value for i, value in enumerate(feature_matrix_x.flatten())}
            predicted_class = model.predict_one(feature_dict)
            probabilities = model.predict_proba_one(feature_dict)
        elif hasattr(model, 'predict'):
            # Use Scikit-learn or general Pipeline method
            # Ensure `data` is transformed appropriately for batch input
            data = [list(data.values())]  # Convert dict to a list of values
            predicted_class = model.predict(data)[0]
            probabilities = model.predict_proba(data)[0]
        else:
            raise Exception("Unknown model type: Prediction method not found.")

        # Return the results
        return {"predicted_class": predicted_class}

    # except FileNotFoundError:
    #     raise Exception("Model not found. Prediction is unavailable.")
    # except Exception as e:
    #     raise Exception(f"An error occurred during prediction: {str(e)}")

