# One-Page Report: California Housing Regression

## Motivation

The earlier version used synthetic housing data, so the result was too easy. We replaced it with the real California Housing dataset from scikit-learn.

## Dataset

The dataset contains district-level California housing features and median house value. Features include median income, house age, average rooms, average bedrooms, population, occupancy, latitude, and longitude.

## Method

We compared Ridge regression on original features, Ridge regression with engineered features, and a random forest with engineered features. Engineered features included rooms per bedroom, population per room, and a coastal distance proxy.

## Results

On the holdout set, Ridge base features reached R2 0.5911, engineered Ridge reached R2 0.6101, and engineered random forest reached R2 0.8082. In 5-fold cross-validation, engineered random forest reached mean R2 0.8025 with standard deviation 0.0084.

## Interpretation

Feature engineering slightly helped the linear model. The random forest performed much better because the housing problem contains nonlinear effects, especially geography and income interactions.

## Conclusion

The project now reports credible real-data regression results. The best model is the engineered random forest, and cross-validation confirms that its improvement is stable.
