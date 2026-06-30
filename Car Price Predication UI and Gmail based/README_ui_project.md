# Car Price Prediction — Interactive Web UI (Gradio)

A machine learning project that predicts the resale price of used cars and shows the result **instantly on screen** through a live, shareable web interface — no forms, no email, no waiting.

## Overview

This project trains the same regression model as the email-based version, but instead of routing predictions through Google Forms/Sheets/email, it wraps the trained model in a **Gradio** web interface. Users fill in their car's details directly on a webpage and get the predicted price (with a margin of error) back within seconds, on the same screen.

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

- **Polynomial Features (degree=2)** capture non-linear depreciation patterns (e.g. price drops steeply in the first few years, then levels off).
- **Ridge Regression** (`alpha=10`) controls overfitting on the larger polynomial feature space.
- `include_bias=False` avoids a duplicate intercept, since Ridge fits its own.

### 6. Model Selection — SSE vs Complexity
Polynomial degrees 1–4 were compared on Sum of Squared Errors (SSE):

| Degree | Train SSE | Test SSE | Verdict |
|---|---|---|---|
| 1 | 446.92 | 105.93 | Underfitting |
| **2** | 370.21 | **94.00** | **Best — sweet spot** |
| 3 | 305.78 | 95.55 | Slight overfitting begins |
| 4 | 266.10 | 201.93 | Clear overfitting |

**Degree = 2** was selected as the final model.

### 7. Final Model
- `PolynomialFeatures(degree=2, include_bias=False)` + `Ridge(alpha=10)`.
- Saved to disk with `joblib`: `car_price_model.pkl`, `polynomial_features.pkl`, `scaler.pkl`.

## End-to-End Delivery: Live Gradio Interface

Instead of Google Forms + email, prediction happens entirely in-app:

1. **Load saved model** — `final_model`, `final_poly`, and `scaler` are loaded back from their `.pkl` files via `joblib.load()`, so the app doesn't need to retrain anything at runtime.
2. **`predict_price_ui()` function** — takes the 7 raw inputs (`name, year, km_driven, fuel, seller_type, transmission, owner`) directly as arguments and runs them through the exact same preprocessing pipeline used in training: brand extraction → one-hot encoding → column alignment (`reindex` against training columns) → scaling → polynomial expansion → prediction → `expm1` to convert back from log-price to rupees.
3. **Gradio Interface** (`gr.Interface`) maps each input field to a UI widget:
   - `Textbox` for Car Name
   - `Number` for Year and Km Driven
   - `Dropdown` for Fuel, Seller Type, Transmission, Owner (restricts input to categories the model was actually trained on, preventing typos that would break one-hot encoding)
4. **`interface.launch(share=True)`** starts a local server inside Colab and also generates a **public shareable link** (valid 72 hours) — anyone with the link can use the tool without needing Colab access themselves.
5. Output is shown as: `₹X (Range: ₹(X-RMSE) – ₹(X+RMSE))`, using the same ±RMSE margin approach as the email version for an honest uncertainty range.

## Tech Stack
- **Data:** pandas, numpy
- **Modeling:** scikit-learn (`LinearRegression`, `Ridge`, `PolynomialFeatures`, `StandardScaler`)
- **Visualization:** matplotlib
- **Interface:** Gradio
- **Persistence:** `joblib`
- **Environment:** Google Colab

## Why Gradio Instead of Forms/Email
| | Email-based | Gradio UI |
|---|---|---|
| User collects | Email + 7 fields | 7 fields only (no email needed) |
| Result delivery | Async, via email | Instant, on the same page |
| Re-run needed? | Yes, to check new responses | No, runs live |
| Shareability | Form link | Public Gradio link (72hr) |

## Known Limitations
- The public `share=True` link expires after 72 hours and only stays live while the Colab cell is running — closing the notebook or runtime ends the session.
- Dataset lacks `engine`, `mileage`, `max_power`, and `torque` columns, so predictions rely only on basic listing attributes.
- `car_age` is hardcoded against the year 2024 rather than computed dynamically from the current date.
- No input validation beyond dropdown restrictions — e.g. a `Year` far in the future or negative `Km Driven` isn't explicitly blocked.
- For a persistent, always-on deployment (rather than a temporary Colab-hosted link), this would need to be moved to a proper hosting service (e.g. Hugging Face Spaces, a cloud VM, or similar).
