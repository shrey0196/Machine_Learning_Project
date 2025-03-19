**Titanic Survival Prediction using Logistic Regression**

Overview

This project predicts passenger survival on the Titanic using a Logistic Regression model. The dataset undergoes data cleaning, feature engineering, and model evaluation to determine survival probabilities.

Technologies Used

Python

Pandas

NumPy

Matplotlib & Seaborn (for visualization)

Scikit-Learn (for machine learning)

Cufflinks (for interactive visualizations)

Dataset

We use the Titanic dataset from Kaggle, which contains details about passengers, including age, gender, ticket class, and survival outcome.

Data Preprocessing

Handle Missing Values

Fill missing Age values based on passenger class.

Drop the Cabin column due to excessive missing values.

Feature Engineering

Convert categorical variables (Sex, Embarked) into numerical values.

Drop irrelevant columns (Name, Ticket, PassengerId).

Results

The classification report provides precision, recall, and F1-score.

The confusion matrix visualizes correct and incorrect predictions.

Model performance can be improved with feature selection and hyperparameter tuning.
