from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

def tune_model():
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]}
    model = HistGradientBoostingClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X, y) # Assuming X, y are loaded appropriately
    joblib.dump(grid_search.best_estimator_, 'ml_app/ml/model.pkl')