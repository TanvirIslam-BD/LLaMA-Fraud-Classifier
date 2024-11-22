import joblib
from numpy.random import Generator, PCG64

from ml_app.ml.train import prepare_json_data_to_prediction

rng = Generator(PCG64())
print(rng.random())

def prediction(data):
    try:

        # Load the saved model
        model = joblib.load('ml_app/saved_models/model.pkl')

        # Preprocessing the json data
        data, feature_matrix_x = prepare_json_data_to_prediction(data)

        # Check if it's a River model or a Pipeline
        if hasattr(model, 'predict_one'):
            # Use River's incremental learning method
            predicted_class = model.predict_one(data)
            probabilities = model.predict_proba_one(data)
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

    except FileNotFoundError:
        raise Exception("Model not found. Prediction is unavailable.")
    except Exception as e:
        raise Exception(f"An error occurred during prediction: {str(e)}")

