# Report: California Housing Regression with Feature Engineering

## Motivation

We studied housing-price regression to test whether feature engineering improves predictive performance.

## Dataset

The dataset is a controlled California-style housing table with income, age, room, population, occupancy, and geographic features. It is not the downloaded California Housing dataset.

## Method

We compared Ridge regression with base features, Ridge regression with engineered features, and Random Forest regression with engineered features.

## Hyperparameters

The test split was 25 percent. Ridge used `alpha=1.0`. Random forest used 200 trees and maximum depth 8.

## Results

Ridge with base features achieved R2 = 0.9150. Ridge with engineered features improved to R2 = 0.9515. Random forest with engineered features achieved R2 = 0.9442.

## Interpretation

Feature engineering improved the result clearly. The new ratio and location features made the regression problem easier for the linear model. Random forest was strong, but engineered Ridge was best.

## Conclusion

The project shows why baselines are necessary: here engineered features did help, and the improvement is visible in the metrics.
