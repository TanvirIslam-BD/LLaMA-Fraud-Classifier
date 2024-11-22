from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV

from ml_app.ml.pipeline import create_pipeline


def model_train_with_auto_tune(X, y, categorical_features, numeric_features):
    # Create the initial pipeline
    pipeline = create_pipeline(categorical_features, numeric_features)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_iter": [100, 200, 300],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__min_samples_leaf": [10, 20, 30]
    }

    # Set up GridSearchCV with cross-validation
    grid_search = RandomizedSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",  # Adjust scoring based on your specific task
        n_jobs=-1,  # Use all available CPUs
        verbose=2
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run the grid search to find the best parameters
    grid_search.fit(X_train, y_train)

    # Best model with optimal hyperparameters
    best_model = grid_search.best_estimator_

    # Evaluation on test set
    test_score = best_model.score(X_test, y_test)

    # Return best model, parameters, and test score
    return best_model, grid_search.best_params_, test_score