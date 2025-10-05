# Pima-Indians-Diabetes-Prediction-Using-Logistic-Regression
The objective of this project is to build a predictive model to determine whether a patient has diabetes based on health measurements. This is a binary classification problem using the Pima Indians Diabetes dataset from Kaggle (UCI dataset).

## Why this dataset:

* Small and clean, perfect for beginners in machine learning.

* Classic dataset for logistic regression.

* Includes 8 predictors (like glucose, BMI, age) and 1 target (Outcome — 0 = non-diabetic, 1 = diabetic).

### 1) Data Understanding (Exploratory Data Analysis)

Why we did this step:
* Before building a model, it’s crucial to understand the data structure, missing values, distributions, and relationships between features and the target.

What we did:

* Checked dataset shape and column types.

* Observed the target balance: 500 non-diabetic, 268 diabetic.

* Checked for impossible zeros in features like Glucose, BloodPressure, BMI, SkinThickness, Insulin.

* Visualized distributions with histograms and boxplots.

* Calculated correlations between features and the target to see which features are likely important.

Key findings:

* Some features had zero values that are biologically impossible → treated as missing.

* Features like Glucose, BMI, and Age show clear differences between diabetic and non-diabetic classes.

### 2) Data Preprocessing

Why we did this step:
* Machine learning models cannot handle missing values and may perform poorly with unscaled features. Preprocessing ensures clean, usable input and improves model performance.

What we did:

* Replaced zeros in specific columns with NaN (treating them as missing values).

* Imputed missing values with the median (robust to outliers).

* Split dataset into training and test sets (80/20 split), ensuring class balance with stratification.

* Scaled features using StandardScaler so all features have similar ranges.

Impact:

* Avoided model errors due to missing values.

* Scaling ensured the logistic regression model converged quickly and coefficients were comparable.

### 3) Baseline Model: Logistic Regression

Why we did this step:
* Logistic regression is simple, interpretable, and suitable for binary classification problems like diabetes prediction. Starting with a baseline helps measure model performance before tuning.

What we did:

* Trained logistic regression on training data.

* Evaluated using classification report (precision, recall, F1-score) and ROC AUC.

* Checked confusion matrix and ROC curve.

Key insights:

* Model performed reasonably well with default parameters.

* True positives (correctly predicted diabetics) and false negatives (missed diabetics) were analyzed via confusion matrix.

### 4) Model Evaluation & Interpretation

Why we did this step:
* Understanding model metrics and feature importance is crucial, especially in medical predictions where recall (catching positives) may be more important than accuracy.

What we did:

* Plotted ROC and Precision-Recall curves to understand performance across thresholds.

* Interpreted coefficients as odds ratios to see which features increase diabetes risk:

* Glucose and BMI have the strongest positive impact.

* Age also contributes moderately.

* Used model coefficients to explain feature importance in plain language.

Impact:

Clinical insight: higher glucose, BMI, and age → higher probability of diabetes.

### 5) Hyperparameter Tuning

Why we did this step:
* Tuning hyperparameters improves model performance and avoids overfitting.

What we did:

* Used GridSearchCV to tune the regularization parameter C.

* Selected the parameter value that maximized ROC AUC via 5-fold cross-validation.

### Result:

* Optimized model performed slightly better than baseline.

* ROC AUC improved, meaning better ranking of predicted probabilities.

## 6) Handling Class Imbalance (Optional Enhancement)

Why we did this step:
* Diabetic cases are fewer than non-diabetic cases. Models may bias toward the majority class.

What we did:

* Tried class weighting in logistic regression (class_weight='balanced').

* Used SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic diabetic samples in the training set.

Impact:

* Improved recall (fewer false negatives).

* Ensured the model does not ignore minority (diabetic) class.

### 7) Final Model & Saving

Why we did this step:
* Saving the trained model allows for reuse without retraining and supports deployment or real-time prediction.

What we did:

* Saved model + scaler using joblib.

* Demonstrated predicting a new patient’s probability of diabetes.

### 8) Key Learnings & Conclusion

Learnings:

* Logistic regression is simple but interpretable.

* Proper EDA and preprocessing (handling zeros, scaling, splitting) is crucial.

* Feature interpretation via odds ratios provides clinical insights.

* Balancing classes improves detection of minority outcomes.

Conclusion:

* The model predicts diabetes with good performance (ROC AUC > 0.80).

* Glucose, BMI, and Age are the most influential features.

* Threshold selection can be tuned depending on whether precision or recall is more important in practice.

The model predicts diabetes with good performance (ROC AUC > 0.80).

Glucose, BMI, and Age are the most influential features.

Threshold selection can be tuned depending on whether precision or recall is more important in practice.

