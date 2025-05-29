# GUT-ML-Project-1
Admission Prediction &amp; Scholarship Recommendation This project uses data science and machine learning techniques to predict the chance of university admission for applicants. It includes exploratory data analysis (EDA), feature engineering, and trains multiple regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor). 

>>>Import Libraries: It starts by importing necessary libraries for data manipulation (pandas, numpy), visualization (matplotlib, seaborn), and machine learning (sklearn).

>>>Upload Dataset: It mounts the user's Google Drive to access the dataset and then reads a CSV file named "Admission_Predict.csv" into a pandas DataFrame.
>>>Data Exploration (EDA - Exploratory Data Analysis):
>>>Checks the shape of the DataFrame (number of rows and columns).
>>>Displays the first few rows to understand the data structure.
>>>Gets information about the DataFrame, including data types and non-null counts.
>>>Calculates the sum of null values for each column to check for missing data.
>>>Uses value_counts() to analyze the distribution of unique values in specific columns ('SOP', 'Research', 'CGPA', 'LOR') to understand if they are continuous or categorical.
>>>Provides a statistical overview of the numerical columns using describe().
>>>Discusses the insights from the statistical summary, focusing on missing values and potential skewness/outliers based on mean/max comparison.
>>>Drops the 'Serial No.' column as it's not relevant for the analysis.
>>>Checks for duplicate rows and explains the difference between df.duplicated() and df.duplicated().sum().
>>>Visualizes the distribution of numerical features ('GRE Score', 'TOEFL Score', 'CGPA') using box plots to detect outliers.
>>>Analyzes outliers based on domain knowledge and decides whether to keep them.
>>>Visualizes the distribution of categorical features ('University Rating', 'SOP', 'LOR', 'Research', 'Chance of Admit') using box plots and discusses the findings.
>>>Explains the process of outlier filtration using the IQR method for 'LOR' and 'CGPA' and prints the identified outliers.
>>>Discusses checking for data inconsistencies using methods like unique().
>>>Displays unique values for selected columns ('GRE Score', 'CGPA', 'Research') to check for inconsistencies.
>>>Visualizes the distribution of all numerical features using histograms.
>>>Discusses the distribution patterns observed in the histograms (normally distributed, skewed, discrete).
>>>Creates scatter plots for each feature against the 'Chance of Admit' target variable to visualize relationships.
>>>Explains the purpose of scatter plots and how to interpret them.
>>>Generates a scatter matrix (pair plot) using seaborn to visualize relationships between all pairs of numerical columns.
>>>Explains the purpose and benefits of a scatter matrix in the EDA process.
Feature Engineering:
>>>Creates a correlation heatmap to visualize the correlation between all pairs of numerical columns, highlighting relationships between input features and the target variable.
>>>Explains the purpose of a correlation heatmap.
>>>Demonstrates Min-Max Scaling (Normalization) to scale features to a specific range (0-1) and visualizes the histograms after scaling.
>>>Explains the process of Standard Scaling (Z-score Normalization) but only fits the scaler without transforming the data for demonstration.
>>>Model Training:
>>>Imports necessary modules for model selection, regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor), and evaluation metrics.
>>>Defines the features (input variables) and the target variable ('Chance of Admit').
>>>Splits the data into training and testing sets using train_test_split.
>>>Trains a Linear Regression model on the training data and makes predictions on the test data.
>>>Trains a Decision Tree Regressor model with a specified max_depth on the training data and makes predictions.
>>>Trains a Random Forest Regressor model with a specified number of estimators on the training data and makes predictions.
>>>Evaluation Metrics:
>>>Calculates and prints the Mean Squared Error (MSE) and R-squared score for each trained model to evaluate their performance.
>>>Feature Importance:
>>>Calculates and visualizes the feature importance from the trained Random Forest model using a horizontal bar plot, showing which features have the most impact on the prediction.
>>>Decision Tree Visualization:
>>>Visualizes the trained Decision Tree model using plot_tree, providing a visual representation of the decision rules learned by the model.
>>>Hypothetical Applicant Prediction:
>>>Creates a pandas DataFrame representing a hypothetical applicant's data.
>>>Uses each trained model (Linear Regression, Decision Tree, Random Forest) to predict the 'Chance of Admit' for this hypothetical applicant.
>>>Prints the predicted probabilities from each model.
>>>Scholarship Recommendation Logic:
>>>Implements a simple logic based on the Random Forest model's prediction and the applicant's CGPA to recommend for a scholarship.
>>>Model Saving:
>>>Saves the trained Random Forest model to a pickle file named "my_modelGTU.pkl" in a specified directory within the user's Google Drive.
>>>In essence, the code aims to build and evaluate machine learning models to predict the chance of admission based on various applicant attributes and then uses the best-performing model (Random Forest) to demonstrate a hypothetical prediction and a simple scholarship recommendation logic.
