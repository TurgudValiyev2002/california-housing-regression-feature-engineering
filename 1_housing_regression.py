from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RESULTS = Path("results")
RANDOM_STATE = 42


def load_real_data() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.rename(columns={"MedHouseVal": "MedHouseVal"})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RoomsPerBedroom"] = out["AveRooms"] / out["AveBedrms"].replace(0, np.nan)
    out["PopulationPerRoom"] = out["Population"] / out["AveRooms"].replace(0, np.nan)
    out["CoastalDistanceProxy"] = np.sqrt((out["Longitude"] + 121) ** 2 + (out["Latitude"] - 37) ** 2)
    return out.replace([np.inf, -np.inf], np.nan).fillna(out.median(numeric_only=True))


def evaluate_holdout(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "model": name,
        "mae": round(mean_absolute_error(y_test, pred), 4),
        "rmse": round(mean_squared_error(y_test, pred) ** 0.5, 4),
        "r2": round(r2_score(y_test, pred), 4),
    }


def cross_validation_table(model_specs: list[tuple[str, object, pd.DataFrame]], y) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model, x in model_specs:
        scores = cross_validate(
            model,
            x,
            y,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=None,
        )
        rows.append(
            {
                "model": name,
                "cv_mae_mean": round(-scores["test_mae"].mean(), 4),
                "cv_mae_std": round(scores["test_mae"].std(), 4),
                "cv_r2_mean": round(scores["test_r2"].mean(), 4),
                "cv_r2_std": round(scores["test_r2"].std(), 4),
            }
        )
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load_real_data()
    engineered = add_features(df)
    df.to_csv(RESULTS / "california_housing_real_data_sample.csv", index=False)
    engineered.to_csv(RESULTS / "engineered_housing_data.csv", index=False)

    y = engineered["MedHouseVal"]
    base_x = df.drop(columns=["MedHouseVal"])
    eng_x = engineered.drop(columns=["MedHouseVal"])
    xb_train, xb_test, y_train, y_test = train_test_split(base_x, y, test_size=0.25, random_state=RANDOM_STATE)
    xe_train, xe_test, _, _ = train_test_split(eng_x, y, test_size=0.25, random_state=RANDOM_STATE)

    base_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    engineered_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    rf_model = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)

    holdout_rows = [
        evaluate_holdout("ridge_base_features", base_model, xb_train, xb_test, y_train, y_test),
        evaluate_holdout("ridge_engineered_features", engineered_model, xe_train, xe_test, y_train, y_test),
        evaluate_holdout("random_forest_engineered", rf_model, xe_train, xe_test, y_train, y_test),
    ]
    holdout_metrics = pd.DataFrame(holdout_rows)
    holdout_metrics.to_csv(RESULTS / "regression_metrics.csv", index=False)

    cv_metrics = cross_validation_table(
        [
            ("ridge_base_features", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]), base_x),
            ("ridge_engineered_features", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]), eng_x),
            (
                "random_forest_engineered",
                RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
                eng_x,
            ),
        ],
        y,
    )
    cv_metrics.to_csv(RESULTS / "cross_validation_metrics.csv", index=False)

    rf_fitted = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1).fit(xe_train, y_train)
    imp = pd.DataFrame({"feature": xe_train.columns, "importance": rf_fitted.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(RESULTS / "feature_importance.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(holdout_metrics["model"], holdout_metrics["r2"], color=["#3d6fb6", "#4a8f5a", "#b26a3b"])
    plt.ylabel("Holdout R2 score")
    plt.title("Real California Housing Regression")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "r2_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    top = imp.head(8).sort_values("importance")
    plt.barh(top["feature"], top["importance"], color="#4a8f5a")
    plt.title("Top Housing Features")
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=180)
    plt.close()

    print(holdout_metrics.to_string(index=False))
    print(cv_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
