# California Housing Regression with Feature Engineering

## Motivation

Housing-price prediction is a useful regression problem because it combines numeric features, geographic signals, and feature engineering. The goal is not only to predict well, but also to understand which variables explain the prediction.

## Project Goal

We compared baseline regression with engineered features and tested whether feature engineering improves prediction.

## Dataset

We used a controlled California-style housing dataset with features similar to the common California Housing dataset: median income, house age, rooms, bedrooms, population, occupancy, latitude, and longitude. It is a local proxy dataset so the project runs without external downloads.

## Tools

Python, NumPy, pandas, scikit-learn, and matplotlib.

## Method

We created three additional features: rooms per bedroom, population per room, and a coastal-distance proxy. We compared Ridge regression on base features, Ridge regression on engineered features, and Random Forest regression on engineered features.

## Hyperparameters

- Test split: 25 percent
- Ridge: `alpha=1.0`
- Random Forest: `n_estimators=200`, `max_depth=8`, `random_state=42`

## Results

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Ridge base features | 0.2791 | 0.3468 | 0.9150 |
| Ridge engineered features | 0.2047 | 0.2619 | 0.9515 |
| Random forest engineered | 0.2205 | 0.2810 | 0.9442 |

Result files include the base data, engineered data, regression metrics, feature importance, and figures.

## Interpretation

Ridge regression with engineered features performed best. The engineered coastal-distance and ratio features helped the linear model capture structure that was less direct in the base features. Random forest also performed well, but it was slightly weaker than engineered Ridge on this dataset.

## Conclusion

Feature engineering was useful in this experiment. The important lesson is that we should test engineered features against a baseline and report the actual improvement instead of assuming they help.

## How To Run

```bash
pip install -r requirements.txt
python 1_housing_regression.py
```
