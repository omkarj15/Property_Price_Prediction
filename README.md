# Property Price Prediction

## Introduction
Property Price Prediction is crucial for companies like 99acres and Magic Bricks. Understanding factors like property age and area helps in efficient resource allocation and decision-making.

## Objective
The objective is to develop a predictive model leveraging machine learning and deep learning algorithms for accurate property value prediction.

## Process Flow
1. Topic Selection
2. Dataset Search
3. Data Pre-processing & Visualization
4. Algorithm Application
5. Model Building
6. Interpretation & Conclusions

## Tools & Platform Used
- Tools: Python
- Platform: Jupyter Notebook, AWS, Visual Studio Code
- Libraries: Sklearn, Pandas, NumPy, Seaborn, Matplotlib, NLTK, TensorFlow

## Data Pre-processing & Visualization
- Data Description
- Handling Null and Duplicate Values
- Outlier Treatment
- Data Cleaning
- Data Visualization
- One-Hot-Encoding for Categorical Variables
- Data Partition

### Details of Dataset
- 38,502 entries & 85 columns
- 32 categorical and 53 numerical columns
- No duplicate values
- Outliers detected using Boxplot
- Missing values in specific columns filled with median due to outliers

## NLP Analysis
- Term Document Matrix (TDM) for description & secondary tags
- Sentiment Analysis for description & secondary tags
- Conversion of Categorical Variables into Dummy Variables

## Model (Machine Learning)
- Linear Regression
- Ridge Regression
- Lasso Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Adaptive Boosting
- XGBoost

## Model (Deep Learning)
- Artificial Neural Network (ANN)
  - **Cost Function**: Mean Absolute Error
    - Train: 4.92
    - Test: 5.19
  - **Accuracy**: R2 Score
    - Train: 83%
    - Test: 82%

## Conclusion
1. XGBoost demonstrates lower RMSE and higher R2 value compared to other ML algorithms.
2. ANN model exhibits lower cost function and good accuracy.

