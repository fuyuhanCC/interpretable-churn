# Interpretable Customer Churn Prediction

This project explores **customer churn prediction** using interpretable
machine learning techniques. Instead of focusing solely on predictive
performance, the emphasis is placed on **model transparency and explainability**,
making the results suitable for analysis, discussion, and decision support.

The project is designed as a practical, end-to-end AI/ML pipeline implemented
in Python.

---

## 📂 Project Structure

interpretable-churn/  
├── data/  
│ └── raw/  
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv  
├── notebooks/  
│ ├── 01_eda.ipynb  
│ ├── 02_modeling.ipynb  
│ └── 03_interpretation.ipynb  
├── requirements.txt  
├── .gitignore  
└── README.md  


---

## 🧠 Project Overview

Customer churn is a critical problem in many industries, particularly in
subscription-based services. This project aims to:

- Explore and understand customer behavior through **exploratory data analysis**
- Build a **baseline churn prediction model**
- Compare model performance with a focus on interpretability
- Use **SHAP (SHapley Additive exPlanations)** to explain model predictions

---

## 📊 01 – Exploratory Data Analysis (EDA)

The EDA phase focuses on understanding the structure and characteristics of
the dataset:

- Distribution of churn vs. non-churn customers
- Analysis of numerical features such as tenure and monthly charges
- Exploration of categorical variables including contract type and services
- Identification of patterns correlated with churn behavior

---

## 🤖 02 – Modeling

A supervised learning pipeline is implemented using scikit-learn:

- Data preprocessing with:
  - Standard scaling for numerical features
  - One-hot encoding for categorical features
- Models explored:
  - Logistic Regression (primary baseline)
  - Decision Tree (for comparison)
- Evaluation metric:
  - ROC-AUC score

The logistic regression model achieves strong performance while maintaining
high interpretability, making it suitable for further explanation.

---

## 📊 03 – Explainability & Model Interpretation

To ensure transparency and trustworthiness of the churn prediction model,
we focus on **interpretable machine learning** rather than black-box accuracy
alone.

A logistic regression model was selected as the primary baseline due to its
inherent interpretability. To further analyze feature contributions, we apply
**SHAP (SHapley Additive exPlanations)**, which provides consistent, additive
feature attributions grounded in cooperative game theory.

### Global Explanations

Global SHAP analysis highlights the most influential factors driving customer
churn across the dataset:

- **Contract type** (especially month-to-month contracts) is the strongest
  driver of churn risk.
- **Monthly charges** have a positive contribution to churn probability,
  indicating price sensitivity among customers.
- **Customer tenure** shows a strong negative effect, suggesting that longer
  relationships reduce churn likelihood.
- Optional services such as online security and technical support contribute
  to customer retention.

These findings align with domain intuition and confirm that the model captures
meaningful behavioral patterns rather than spurious correlations.

### Methodological Note

SHAP explanations are computed using an **Independent masker** based on the
training data distribution. This assumes conditional independence between
features after preprocessing, providing a reasonable trade-off between
interpretability and computational efficiency for linear models.

---

## ⚙️ Environment & Reproducibility

- Python 3.11
- Key libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - shap

To set up the environment:

```bash
pip install -r requirements.txt
```


🚀 Future Work

Explore non-linear but interpretable models (e.g. GAMs)

Extend local explanations for individual customer cases

Evaluate fairness and bias across customer subgroups

Compare SHAP with alternative explainability methods

📌 Notes

This project is intended as a practical AI/ML case study with an emphasis
on explainability and clarity rather than production deployment.