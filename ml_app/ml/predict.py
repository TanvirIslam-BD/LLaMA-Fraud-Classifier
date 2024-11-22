import joblib

def prediction(data):
    try:
        model = joblib.load('ml_app/saved_models/model.pkl')
        return model.predict(data)
    except FileNotFoundError:
        raise Exception("Model not loaded. Prediction is unavailable.")

