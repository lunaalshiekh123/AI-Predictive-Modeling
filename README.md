# Customer Satisfaction Prediction Model 🚀

This project implements a **Machine Learning Classification Pipeline** to predict customer satisfaction levels based on e-commerce shipping and product data.

## 📊 Project Overview
The goal is to classify customers into three categories: **Unhappy, Neutral, and Very Good**, based on features like cost, shipping mode, and discounts offered.

## 🛠️ Technical Features:
- **Feature Engineering:** Transformed raw customer ratings into meaningful satisfaction categories.
- **Automated Pipeline:** Utilized `ColumnTransformer` and `Pipeline` for a seamless flow of data from preprocessing to prediction.
- **Data Preprocessing:** 
    - Scaled numerical features using `StandardScaler`.
    - Encoded categorical variables using `OneHotEncoder`.
- **Handling Imbalanced Data:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns fairly from all classes.
- **Hyperparameter Tuning:** Optimized the **Logistic Regression** model using `GridSearchCV` with 5-fold cross-validation.

## 🧪 Tools & Libraries:
- **Python** (Core Programming)
- **Pandas** (Data Manipulation)
- **Scikit-learn** (Machine Learning & Pipelines)
- **Imbalanced-learn** (SMOTE for class balancing)
