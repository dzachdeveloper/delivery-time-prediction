"""Train candidate models, track them in MLflow, register the best one as @production."""
import inspect
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data import FEATURES, TARGET, make_data

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "delivery-eta"
MODEL_NAME = "delivery-eta-model"

# TODO 2a: TOLERANCE_MIN. We also report the share of orders predicted within +/- this many minutes.
#   Option A: 3   -> strict, few predictions count as "good"
#   Option B: 5   -> what most customers would accept
#   Option C: 10  -> generous
# THINK: A late order earns a complaint, a refund or a lost customer; an over-cautious promise makes customers choose another restaurant. How late is "too late" for your customers?
TOLERANCE_MIN = 5

# TODO 2b: quality gate. The best model is NOT registered if its MAE (minutes) is ABOVE this value.
#   Option A: 5.0 (lenient)   Option B: 4.0   Option C: 3.0 (strict)
# HINT: the lazy model from Part 1 is off by about 7.7 minutes.
MAX_MAE = 4.0

# TODO 2e: which metric picks the winner? (used further down)
#   Option A: "mae"              -> average minutes off (lower is better), easy to explain
#   Option B: "rmse"             -> like MAE but punishes big misses more (lower is better)
#   Option C: "within_tolerance" -> share of orders within TOLERANCE_MIN (higher is better)
# THINK: one 30-minute miss or ten 3-minute misses: which hurts the business more?
SELECT_BY = "rmse"

# Newer MLflow saves sklearn models with skops (safer than pickle). skops only loads object types it has
# been told to trust. Tree-based models contain "sklearn.tree._tree.Tree" objects, which are safe here
# because this script created them. If you add a model and get an error saying
# "Untrusted types found in the file: [...]", add the type named in that error to this list.
# THINK: why would you NOT want to trust every type automatically? (security slide!)
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]
SAVE_OPTIONS = {}
if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
    SAVE_OPTIONS["skops_trusted_types"] = TRUSTED_TYPES


def check_todos():
    missing = [name for name, value in [("TOLERANCE_MIN (TODO 2a)", TOLERANCE_MIN), ("MAX_MAE (TODO 2b)", MAX_MAE),
                                        ("SELECT_BY (TODO 2e)", SELECT_BY)] if value is None]
    if missing:
        raise SystemExit("Fill these in before training: " + ", ".join(missing))


def main():
    check_todos()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42)

    candidates = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
        # TODO 2c (recommended): add a third candidate model. Pick ONE and uncomment it by deleting the
        #   "#   Option X: " part at the start of its line. Then compare all models in MLflow.
        #   Option A: "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        #   Option B: "extra_trees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=42),
        #   Option C: "gradient_boosting": GradientBoostingRegressor(random_state=42),
        #   Option D: "knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
        # THINK: linear regression gives one number per feature ("+2.6 min per km"). Is that worth a bit of accuracy?
        "gradient_boosting": GradientBoostingRegressor(random_state=42),        
    }

    results = {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)                       # predicted minutes
            within = float((abs(y_test - pred) <= TOLERANCE_MIN).mean())

            # TODO 2d: compute MAE and RMSE from y_test and pred.
            # HINT: mean_absolute_error(y_test, pred)   and   np.sqrt(mean_squared_error(y_test, pred))
            # THINK: explain MAE to a restaurant owner in one sentence.
            metrics = {
                "mae": mean_absolute_error(y_test, pred),
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "r2": r2_score(y_test, pred),
                "within_tolerance": within,
            }
            if metrics["mae"] is None or metrics["rmse"] is None:
                raise SystemExit("TODO 2d: fill in mae and rmse")
            mlflow.log_params({"model_type": name, "tolerance_min": TOLERANCE_MIN, "n_rows": len(df)})
            mlflow.log_metrics(metrics)
            info = mlflow.sklearn.log_model(model, name="model", input_example=X_test.head(3), **SAVE_OPTIONS)
            results[name] = {"metrics": metrics, "uri": info.model_uri, "model": model}
            print(f"{name:20s}", {k: round(v, 3) for k, v in metrics.items()})

    lower_is_better = SELECT_BY in ("mae", "rmse")
    pick = min if lower_is_better else max
    best = pick(results, key=lambda n: results[n]["metrics"][SELECT_BY])
    best_metrics = results[best]["metrics"]
    print(f"\nWinner by {SELECT_BY}: {best}")
    if best_metrics["mae"] > MAX_MAE:
        raise SystemExit(f"Quality gate FAILED: {best} has MAE {best_metrics['mae']:.2f} min > {MAX_MAE}")

    # Save the model file first: the API tests and the Docker image use it
    joblib.dump(results[best]["model"], "model.joblib")
    print("Saved model.joblib")

    # TODO 2f: register the winner in the MLflow Model Registry and point the alias "production" at it.
    # HINT:  version = mlflow.register_model(results[best]["uri"], MODEL_NAME).version
    #        MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version)
    # THINK: the API will load "models:/delivery-eta-model@production". Why is an alias better than
    #        writing a version number into the API code? How would you roll back?
    version = mlflow.register_model(results[best]["uri"], MODEL_NAME).version
    MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version)
    if version is None:
        raise SystemExit("TODO 2f: register the model and set the production alias")

    with open("metrics.json", "w") as f:
        json.dump({"best_model": best, "selected_by": SELECT_BY, "tolerance_min": TOLERANCE_MIN, "max_mae": MAX_MAE,
                   "registered_version": version, **best_metrics}, f, indent=2)
    print(f"Registered {MODEL_NAME} v{version} as @production")


if __name__ == "__main__":
    main()
