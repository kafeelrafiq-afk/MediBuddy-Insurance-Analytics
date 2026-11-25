# MediBuddy Insurance Analytics – Capstone Project

## 1. Project Overview

This project analyses MediBuddy insurance data using **Microsoft Power BI, Python and Machine Learning**.

The objective is to understand patterns in recorded insurance charges, identify important customer risk factors and develop a machine-learning model to predict insurance charges.

## 2. Dataset Description

Two Excel datasets were integrated using an Inner Join on the common column `Policy no.`.

- Total merged records: **1,338**
- Numeric features: Age, BMI and Children
- Categorical features: Gender, Smoking Status and Region
- Target variable: `charges in INR`

## 3. Tools and Technologies

- Microsoft Power BI
- Python
- Pandas
- NumPy
- Scikit-learn
- Excel
- Jupyter Notebook / Python environment
- GitHub

## 4. Business Analysis

The project investigates eight business questions:

1. Does gender matter when extending insurance policies?
2. What is the average amount spent per policy?
3. Should policies differ based on geographic location?
4. Do dependents affect insurance charges?
5. Does BMI provide insights into insurance charges?
6. Is smoking status an important factor?
7. Does age influence insurance charges?
8. Can wellness incentives be considered based on BMI analysis?

## 5. Power BI Dashboard

The interactive dashboard includes:

- Dashboard Overview
- Risk Analysis
- Gender Analysis
- Business Insights
- Business Recommendations
- Key Findings
- BMI Discount Analysis
- Machine Learning Model Performance
- Feature Importance
- Actual vs Predicted Analysis
- Overall Conclusion
- Project Methodology

Interactive charts, KPI cards, slicers and filters are used to explore the data.

## 6. Machine Learning Approach

Three regression models were developed and evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Tuned Random Forest Regressor

### Model Development

- 80/20 training and testing split
- Data preprocessing using Scikit-learn pipelines
- Numerical and categorical feature preprocessing
- GridSearchCV hyperparameter tuning
- Five-fold cross-validation
- Model evaluation using R², MAE and RMSE

GridSearchCV evaluated 108 hyperparameter combinations across five folds.

## 7. Model Performance

**Best-performing model: Tuned Random Forest**

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Linear Regression | 4,181.19 | 5,796.28 | 78.36% |
| Random Forest | 2,534.46 | 4,598.32 | 86.38% |
| Tuned Random Forest | **2,487.92** | **4,349.61** | **87.81%** |

The Tuned Random Forest achieved the best performance among the three evaluated models.

## 8. Key Findings

- Smoking status has the strongest combined feature importance in the model.
- BMI and age also contribute to model predictions.
- Geographic regions show differences in average recorded insurance charges.
- Gender representation is nearly equal across the dataset.
- BMI analysis can help explore preventive healthcare and wellness initiatives.

Feature importance indicates predictive contribution, not cause and effect.

## 9. Project Deliverables

- Final Power BI dashboard (.pbix)
- Professional Python machine-learning script
- Model comparison results (.csv)
- Feature importance results (.csv)
- Cross-validation results (.csv)
- Project documentation PDF

## 10. Conclusion

This project demonstrates how business intelligence and machine learning can be combined to analyse insurance data and generate meaningful insights.

The Tuned Random Forest model achieved an R² score of 87.81% on the reported test evaluation.

The findings represent associations within the available dataset and should not be interpreted as proof of causation. Model predictions are estimates, not guaranteed insurance charges.

---

**Project:** MediBuddy Insurance Analytics  
**Type:** Data Analytics and Machine Learning Capstone Project  
**Tools:** Power BI | Python | Scikit-learn | Excel
