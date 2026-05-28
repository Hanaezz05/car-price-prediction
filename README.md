# Car Price & FIFA Player ML Project

A machine learning project across two datasets — car prices and FIFA player valuations — covering the full pipeline from EDA to ensemble models.

---

## Datasets
| Dataset | Target |
|---|---|
| `car_price.csv` | Car price (regression + classification) |
| `Fifa.csv` | Player market value & performance tier |

---

## Assignment 1 — Car Price Prediction

**Models:** Simple Linear Regression, Multiple Linear Regression, KNN Classifier

**Best Results:**
| Model | Score |
|---|---|
| Multiple Linear Regression | R² = 0.83 |
| KNN Classifier (k=7, Manhattan) | Accuracy = 87% |

**Key finding:** Removing scaling from KNN dropped accuracy from 87% → 73.6%, proving StandardScaler is essential for distance-based models.

---

## Assignment 2 — FIFA Player Valuation

**Models:** Polynomial Regression, Ridge, Lasso, Logistic Regression, Naïve Bayes

**Best Results:**
| Task | Model | Score |
|---|---|---|
| Regression | Ridge (deg 4) | R² ≈ 0.96, RMSE = 0.224 |
| Classification | Logistic Regression (L2) | Accuracy ≈ 77.8% |

**Key finding:** Ridge outperformed Lasso because the dense feature space benefits from keeping all features, not zeroing them out.

---

## Assignment 3 — Advanced Models & Ensembles

**Models:** KNN, SVR, Random Forest → Voting, Gradient Boosting, Stacking

**Best Results:**
| Task | Model | Score |
|---|---|---|
| Regression | SVR (RBF, C=10) | R² = 0.96 |
| Classification | Ensemble (best) | Accuracy > 87% |

**Key finding:** Ensemble methods consistently outperformed individual models by combining complementary error patterns.

---

## Tech Stack
- Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## How to Run
1. Clone the repo
2. Add `car_price.csv` and `Fifa.csv` to the project folder
3. Open `FINAL_PROJECT.ipynb` and run all cells
