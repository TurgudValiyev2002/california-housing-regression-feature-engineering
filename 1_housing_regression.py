from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results")
RNG = np.random.default_rng(42)

def make_data(n=1500):
    med_inc = RNG.gamma(4.0, 1.2, n)
    house_age = RNG.integers(1, 52, n)
    ave_rooms = RNG.normal(5.5, 1.2, n).clip(2, 10)
    ave_bedrooms = (ave_rooms * RNG.normal(0.22, 0.03, n)).clip(0.5, 3)
    population = RNG.integers(100, 5000, n)
    ave_occup = RNG.normal(3.0, 0.8, n).clip(1, 7)
    latitude = RNG.uniform(32, 42, n)
    longitude = RNG.uniform(-124, -114, n)
    coast_score = np.exp(-((longitude + 121) ** 2 + (latitude - 37) ** 2) / 18)
    value = 0.45 * med_inc + 0.04 * ave_rooms - 0.03 * ave_occup + 1.2 * coast_score + 0.003 * house_age + RNG.normal(0, 0.25, n)
    df = pd.DataFrame({
        "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms, "AveBedrms": ave_bedrooms,
        "Population": population, "AveOccup": ave_occup, "Latitude": latitude, "Longitude": longitude,
        "MedHouseVal": value.clip(0.3, None)
    })
    return df

def add_features(df):
    out = df.copy()
    out["RoomsPerBedroom"] = out["AveRooms"] / out["AveBedrms"]
    out["PopulationPerRoom"] = out["Population"] / out["AveRooms"]
    out["CoastalDistanceProxy"] = np.sqrt((out["Longitude"] + 121) ** 2 + (out["Latitude"] - 37) ** 2)
    return out

def eval_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return {"model": name, "mae": round(mean_absolute_error(y_test, pred), 4), "rmse": round(mean_squared_error(y_test, pred) ** 0.5, 4), "r2": round(r2_score(y_test, pred), 4)}

def main():
    RESULTS.mkdir(exist_ok=True)
    df = make_data()
    engineered = add_features(df)
    df.to_csv(RESULTS / "california_style_housing_data.csv", index=False)
    engineered.to_csv(RESULTS / "engineered_housing_data.csv", index=False)
    y = engineered["MedHouseVal"]
    base_x = df.drop(columns=["MedHouseVal"])
    eng_x = engineered.drop(columns=["MedHouseVal"])
    xb_train, xb_test, y_train, y_test = train_test_split(base_x, y, test_size=0.25, random_state=42)
    xe_train, xe_test, _, _ = train_test_split(eng_x, y, test_size=0.25, random_state=42)
    rows = [
        eval_model("ridge_base_features", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]), xb_train, xb_test, y_train, y_test),
        eval_model("ridge_engineered_features", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]), xe_train, xe_test, y_train, y_test),
        eval_model("random_forest_engineered", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42), xe_train, xe_test, y_train, y_test),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "regression_metrics.csv", index=False)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42).fit(xe_train, y_train)
    imp = pd.DataFrame({"feature": xe_train.columns, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(RESULTS / "feature_importance.csv", index=False)
    plt.figure(figsize=(7,4))
    plt.bar(metrics["model"], metrics["r2"], color=["#3d6fb6", "#4a8f5a", "#b26a3b"])
    plt.ylabel("R2 score")
    plt.title("Housing Regression R2")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "r2_comparison.png", dpi=160)
    plt.figure(figsize=(7,4))
    top = imp.head(8).sort_values("importance")
    plt.barh(top["feature"], top["importance"], color="#4a8f5a")
    plt.title("Top Housing Features")
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=160)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
