# Car Price Prediction — Email Delivery (Ridge Regression)

A machine learning project that predicts the resale price of used cars and emails the prediction directly to the user after they submit their car's details through a Google Form.

## Overview

This project trains a regression model on the CarDekho used-car dataset and exposes it through a no-code-frontend workflow: a user fills out a **Google Form**, the response lands in a linked **Google Sheet**, and a Colab notebook reads the new entry, runs it through the trained model, and **emails the predicted price (with a margin of error) back to the user automatically.**

## Dataset

- **Source:** [CarDekho Used Car Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (`CAR DETAILS FROM CAR DEKHO.csv`)
- **Rows / Columns:** ~4,340 rows, 8 columns
- **Columns:** `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`

## Pipeline

### 1. Data Cleaning
- Checked for missing values (none found).
- Removed outliers in `selling_price` and `km_driven` using the **IQR method** (1.5× multiplier), dropping the full row if either column had an extreme value.

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `car_age` | `2024 - year`, more directly useful to a model than raw year |
| `brand` | First word of `name` (with multi-word brand handling, e.g. "Land Rover") |
| `log_selling_price` | `log1p(selling_price)` — target variable, log-transformed to reduce right-skew |
| `km_per_year` | `km_driven / (car_age + 1)` — usage intensity, normalized by age |

### 3. Train/Test Split
- 80% train / 20% test, `random_state=42`.

### 4. Preprocessing
- **Categorical columns** (`fuel`, `seller_type`, `transmission`, `owner`, `brand`) → one-hot encoded with `pd.get_dummies()`, then `align()`ed between train/test so both have identical columns.
- **Numerical columns** (`year`, `km_driven`, `car_age`, `km_per_year`) → scaled with `StandardScaler` (fit on train only, applied to test).

### 5. Modeling
Two models were trained and compared:

| Model | Train R² | Test R² | Test RMSE |
|---|---|---|---|
| Linear Regression | 0.7382 | 0.7638 | ~₹1,49,000 |
| **Polynomial Regression (degree=2) + Ridge** | 0.7776 | 0.7827 | ~₹1,36,568 |

- **Polynomial Features (degree=2)** were added to capture non-linear relationships (e.g. price doesn't depreciate linearly with age).
- **Ridge Regression** (`alpha=10`) was used instead of plain Linear Regression on the expanded polynomial features to control overfitting from the larger feature space.
- `include_bias=False` was used in `PolynomialFeatures` since Ridge already fits its own intercept — avoids a duplicate, redundant bias column.

### 6. Model Selection — SSE vs Complexity
Polynomial degrees 1–4 were compared on Sum of Squared Errors (SSE) for train and test sets:

| Degree | Train SSE | Test SSE | Verdict |
|---|---|---|---|
| 1 | 446.92 | 105.93 | Underfitting |
| **2** | 370.21 | **94.00** | **Best — sweet spot** |
| 3 | 305.78 | 95.55 | Slight overfitting begins |
| 4 | 266.10 | 201.93 | Clear overfitting |

**Degree = 2** was selected as the final model — lowest test error, and the smallest train/test gap (checked via per-sample MSE rather than raw SSE, since train and test sets differ in size).

### 7. Final Model
- `PolynomialFeatures(degree=2, include_bias=False)` + `Ridge(alpha=10)`.
- Trained on the full preprocessed training set, evaluated once on the held-out test set.
- Saved to disk with `joblib`: `car_price_model.pkl`, `polynomial_features.pkl`, `scaler.pkl`.

## End-to-End Delivery: Google Form → Sheet → Email

1. **Google Form** collects: Email address, Car Name, Year, Km Driven, Fuel, Seller Type, Transmission, Owner.
2. Form responses auto-populate a linked **Google Sheet**.
3. In Colab, `gspread` (authenticated via `google.colab.auth`) reads the sheet into a DataFrame.
4. Each response is run through the same preprocessing pipeline used in training (brand extraction → one-hot encoding → column alignment → scaling → polynomial expansion) before being fed to the saved model.
5. The predicted price is converted back from log-scale (`np.expm1`) into rupees.
6. An email is sent via **Gmail SMTP** (using an App Password, not the account password) containing:
   ```
   Predicted Price: ₹X
   Range: ₹(X - RMSE) – ₹(X + RMSE)
   ```
   The **±RMSE margin** (from the Linear Regression test set) is used as an honest, statistically grounded uncertainty range rather than a single point estimate.

## Tech Stack
- **Data:** pandas, numpy
- **Modeling:** scikit-learn (`LinearRegression`, `Ridge`, `PolynomialFeatures`, `StandardScaler`)
- **Visualization:** matplotlib
- **Delivery:** Google Forms, Google Sheets, `gspread`, `smtplib` (Gmail SMTP)
- **Persistence:** `joblib`
- **Environment:** Google Colab

## Known Limitations
- Dataset lacks `engine`, `mileage`, `max_power`, and `torque` columns (present in the richer `Car details v3.csv` version), so the model relies only on basic listing attributes.
- IQR-based outlier removal drops the entire row if *either* `selling_price` or `km_driven` is an outlier, which can discard otherwise valid data points.
- Email delivery is **not real-time** — the notebook must be manually re-run to check for and process new form responses; running it again on the same responses will re-send duplicate emails unless filtered out.
- `car_age` is hardcoded against the year 2024 rather than the current year dynamically.
