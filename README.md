# MSCS 634 – Lab 4: Regression Analysis with Regularization Techniques (Diabetes Dataset)

**Student:** Sagar Bhetwal  
**Course:** MSCS 634 M20 – Advanced Big Data & Data Mining  
**Lab:** Lab 4: Regression Analysis with Regularization Techniques
**Dataset:** scikit-learn Diabetes Dataset (`sklearn.datasets.load_diabetes`)  

---

## 1. Purpose of This Lab

This lab demonstrates how different regression techniques perform on a real-world health dataset and how **regularization** can improve generalization by controlling model complexity. The work focuses on:

- Building and evaluating:
  - Simple Linear Regression (single feature)
  - Multiple Linear Regression (all features)
  - Polynomial Regression (polynomial feature expansion)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
- Evaluating each model using:
  - Mean Absolute Error (**MAE**)
  - Mean Squared Error (**MSE**)
  - Root Mean Squared Error (**RMSE**)
  - Coefficient of determination (**R²**)
- Visualizing regression performance using:
  - Predicted vs. actual scatter plots
  - (For simple regression) a fitted regression line

---

## 2. Dataset Overview

The Diabetes dataset contains standardized numerical health measurements used to predict a quantitative measure of diabetes disease progression. The notebook verifies basic structure, explores distributions, and confirms the dataset contains **no missing values**, requiring no additional cleaning.

---

## 3. Methods & Implementation Summary

### Step 1 — Data Preparation
- Loaded dataset from `sklearn.datasets`.
- Created a Pandas DataFrame with feature names and the target column.
- Verified shapes, summary statistics, and missing values.
- Visualized target distribution.

### Step 2 — Simple Linear Regression (One Feature)
- Built a baseline model using a single predictor: **BMI** (`bmi`).
- Split into training/testing sets and evaluated performance.
- Visualized predictions and the fitted regression line.
- **Goal:** establish a simple baseline and observe limitations of using one variable.

### Step 3 — Multiple Linear Regression (All Features)
- Trained linear regression using **all features**.
- Evaluated performance on the same metrics.
- Visualized predicted vs. actual values.
- **Outcome:** improved performance compared to single-feature regression, showing that disease progression is influenced by multiple factors.

### Step 4 — Polynomial Regression
- Applied polynomial feature expansion to the **single-feature (BMI)** model.
- Tested multiple polynomial degrees and compared performance.
- Demonstrated how increasing degree can introduce overfitting (worse test performance as complexity grows).

### Step 5 — Regularization: Ridge & Lasso Regression
- Implemented **Ridge** and **Lasso** using a pipeline with:
  - `StandardScaler()` (important for regularization stability)
  - `Ridge(alpha=...)` or `Lasso(alpha=...)`
- Compared multiple alpha values.
- Visualized predictions for selected alpha values.
- Reviewed Lasso coefficients to observe shrinkage and feature selection behavior.

---

## 4. Final Model Comparison (Test Set)

The final comparison table produced in the notebook (sorted by R²) is summarized below:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Lasso Regression (alpha=1.0) | 42.80 | 2824.57 | 53.15 | 0.4669 |
| Ridge Regression (alpha=100.0) | 43.25 | 2858.22 | 53.46 | 0.4605 |
| Multiple Linear Regression (all features) | 42.79 | 2900.19 | 53.85 | 0.4526 |
| Simple Linear Regression (bmi) | 52.26 | 4061.83 | 63.73 | 0.2334 |
| Polynomial Regression (bmi, degree=1) | 52.26 | 4061.83 | 63.73 | 0.2334 |

**Key interpretation:**
- **Lasso (alpha=1.0)** produced the strongest overall performance (highest R² and lowest RMSE among compared models).
- **Ridge (alpha=100.0)** slightly improved generalization compared to ordinary multiple regression by shrinking coefficients.
- Single-feature models (Simple Linear and Polynomial degree 1) performed substantially worse, reinforcing that prediction improves when multiple measurements are used.

---

## 5. Key Insights

- **Using more predictors helps:** Multiple regression outperformed simple regression because the target is influenced by several health variables, not only BMI.
- **Polynomial complexity can hurt:** Increasing polynomial degree can reduce test performance due to overfitting, especially when using only one feature.
- **Regularization improves stability:** Ridge and Lasso reduced overfitting risk by shrinking coefficients. Lasso additionally supports feature selection by driving some coefficients toward zero.
- **Best model in this run:** **Lasso Regression (alpha=1.0)** achieved the best generalization performance on the test set.

---

## 6. Files Included in This Repository

- `*.ipynb` — Jupyter Notebook containing the full implementation, evaluation, and visualizations.
- `README.md` — This report-style summary of the lab.

---

## 7. How to Run

1. Open the notebook in Jupyter Notebook / JupyterLab.
2. Run cells from top to bottom.
3. Ensure the following Python libraries are installed:
   - `numpy`, `pandas`, `matplotlib`
   - `scikit-learn`

Example install command:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 8. Notes / Decisions

- `random_state=42` was used to keep the train/test split reproducible.
- `StandardScaler()` was used with Ridge/Lasso because regularization is sensitive to feature scale.
- Predicted vs. actual plots were included to visually verify model fit and error spread.
