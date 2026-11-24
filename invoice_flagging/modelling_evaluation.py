
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# Train Random Forest with hyperparameter tuning
def train_random_forest(X_train, y_train):

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score, zero_division=0)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        verbose=1,
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation F1 Score:")
    print(grid_search.best_score_)

    return grid_search


# Evaluate classification model
def evaluate_classifier(model, X_test, y_test, model_name):

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    report = classification_report(
        y_test,
        preds,
        zero_division=0
    )

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_score": f1_score(y_test, preds, zero_division=0)
    }
