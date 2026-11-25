"""
MediBuddy Insurance Charges - Professional Machine Learning Workflow

Run this script from the MEDIBUDDY project folder:
    python medibuddy_ml_professional.py

It creates separate *_professional output files and does not overwrite
the original model or Power BI exports.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# 1. File paths and configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PRICE_FILE = BASE_DIR / "Medibuddy Insurance Data Price (1) (2).xlsx"
PERSONAL_FILE = BASE_DIR / "Medibuddy insurance data personal details (1) (2).xlsx"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "charges in INR"
ID_COLUMN = "Policy no."


def regression_metrics(y_true, y_pred):
    """Return common regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    # -----------------------------
    # 2. Load and integrate datasets
    # -----------------------------
    for file_path in (PRICE_FILE, PERSONAL_FILE):
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find {file_path.name}. Keep both Excel files "
                "in the same folder as this Python script."
            )

    price = pd.read_excel(PRICE_FILE)
    personal = pd.read_excel(PERSONAL_FILE)

    if ID_COLUMN not in price.columns or ID_COLUMN not in personal.columns:
        raise KeyError(f"Both Excel files must contain '{ID_COLUMN}'.")

    data = pd.merge(
        price,
        personal,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one"
    )

    required = [ID_COLUMN, *FEATURES, TARGET]
    missing_columns = [col for col in required if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Required columns missing from merged data: {missing_columns}")

    data = data[required].copy()
    data = data.dropna(subset=[TARGET])

    print("\nDATA QUALITY")
    print("Merged dataset shape:", data.shape)
    print("Missing values by column:\n", data.isna().sum())
    print("Duplicate rows:", data.duplicated().sum())
    print("Duplicate policy numbers:", data[ID_COLUMN].duplicated().sum())

    # -----------------------------
    # 3. Separate features and target
    # -----------------------------
    X = data[FEATURES].copy()
    y = pd.to_numeric(data[TARGET], errors="raise")
    policy_ids = data[ID_COLUMN].copy()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        policy_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nDATA SPLIT")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # -----------------------------
    # 4. Preprocessing pipelines
    #    Fit only on training folds to prevent preprocessing leakage.
    # -----------------------------
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, NUMERIC_FEATURES),
            ("categorical", categorical_preprocessor, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    # -----------------------------
    # 5. Baseline Linear Regression
    # -----------------------------
    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    linear_pipeline.fit(X_train, y_train)
    linear_pred = linear_pipeline.predict(X_test)
    linear_scores = regression_metrics(y_test, linear_pred)

    # -----------------------------
    # 6. Standard Random Forest
    # -----------------------------
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_scores = regression_metrics(y_test, rf_pred)

    # -----------------------------
    # 7. Hyperparameter tuning
    #    Refit on lowest cross-validated MAE; also report CV R2.
    # -----------------------------
    tuned_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=tuned_pipeline,
        param_grid=param_grid,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        },
        refit="MAE",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("\nStarting GridSearchCV: 108 parameter combinations x 5 folds.")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    tuned_pred = best_model.predict(X_test)
    tuned_scores = regression_metrics(y_test, tuned_pred)

    best_index = grid_search.best_index_
    best_cv_mae = -grid_search.cv_results_["mean_test_MAE"][best_index]
    best_cv_rmse = -grid_search.cv_results_["mean_test_RMSE"][best_index]
    best_cv_r2 = grid_search.cv_results_["mean_test_R2"][best_index]

    # -----------------------------
    # 8. Compare models
    # -----------------------------
    comparison = pd.DataFrame([
        {"Model": "Linear Regression", **linear_scores},
        {"Model": "Random Forest", **rf_scores},
        {"Model": "Tuned Random Forest", **tuned_scores},
    ])
    comparison.to_csv(BASE_DIR / "medibuddy_model_comparison_professional.csv", index=False)

    cv_summary = pd.DataFrame([{
        "Best CV MAE": best_cv_mae,
        "Best CV RMSE": best_cv_rmse,
        "Best CV R2": best_cv_r2,
        "CV folds": CV_FOLDS,
        "Parameter combinations": len(grid_search.cv_results_["params"]),
        "Best parameters": str(grid_search.best_params_),
    }])
    cv_summary.to_csv(BASE_DIR / "medibuddy_cross_validation_professional.csv", index=False)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(BASE_DIR / "medibuddy_grid_search_results_professional.csv", index=False)

    # -----------------------------
    # 9. Save test predictions with policy identifiers
    # -----------------------------
    predictions = X_test.copy()
    predictions.insert(0, ID_COLUMN, id_test.values)
    predictions["Actual Charges"] = y_test.values
    predictions["Predicted Charges"] = tuned_pred
    predictions["Absolute Error"] = np.abs(y_test.values - tuned_pred)
    predictions.to_csv(BASE_DIR / "medibuddy_predictions_professional.csv", index=False)

    # -----------------------------
    # 10. Feature importance from fitted pipeline
    # -----------------------------
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_rf = best_model.named_steps["model"]
    transformed_names = fitted_preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": transformed_names,
        "Importance": fitted_rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    # Shorten transformed names for readability
    importance["Feature"] = (
        importance["Feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    importance.to_csv(
        BASE_DIR / "medibuddy_feature_importance_professional.csv",
        index=False
    )

    # -----------------------------
    # 11. Save complete fitted pipeline
    # -----------------------------
    model_path = BASE_DIR / "medibuddy_best_pipeline_professional.pkl"
    joblib.dump(best_model, model_path)

    # -----------------------------
    # 12. Report results
    # -----------------------------
    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    print("\nCROSS-VALIDATION RESULTS")
    print(f"Best CV MAE:  ₹{best_cv_mae:,.2f}")
    print(f"Best CV RMSE: ₹{best_cv_rmse:,.2f}")
    print(f"Best CV R2:   {best_cv_r2:.4f}")

    print("\nBEST HYPERPARAMETERS")
    print(grid_search.best_params_)

    print("\nTUNED RANDOM FOREST TEST RESULTS")
    print(f"MAE:  ₹{tuned_scores['MAE']:,.2f}")
    print(f"RMSE: ₹{tuned_scores['RMSE']:,.2f}")
    print(f"R2:   {tuned_scores['R2']:.4f}")

    print("\nTOP FEATURE IMPORTANCE")
    print(importance.head(8).to_string(index=False))

    print("\nFILES CREATED")
    print(model_path.name)
    print("medibuddy_model_comparison_professional.csv")
    print("medibuddy_cross_validation_professional.csv")
    print("medibuddy_grid_search_results_professional.csv")
    print("medibuddy_predictions_professional.csv")
    print("medibuddy_feature_importance_professional.csv")
    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
