# ================================================================
# End-to-end Logistic Regression with preprocessing, evaluation, and plots.
# ================================================================

import os  # For managing folders and file paths
import numpy as np # Efficial numerical computation and arrays
import pandas as pd # Reading csv data and tabular data
import matplotlib.pyplot as plt # Plotting grapths

# scikit-learn: machine learning toolkit for preprocessing, training and evaluating
from sklearn.model_selection import train_test_split  # splitting data into training and test sets
from sklearn.compose import ColumnTransformer # applies different preprocessing steps to columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Converts categories and scales numerical values
from sklearn.pipeline import Pipeline # Chains preprocessing and model steps together
from sklearn.linear_model import LogisticRegression # Main classification model
from sklearn.metrics import (accuracy_score, precision_score, recall_score,  # Functions and measure model perfprmance
                            f1_score, roc_auc_score, roc_curve, precision_recall_curve, 
                            confusion_matrix, classification_report) 

#   Configurations
CSV_PATH = "data/train_and_test2.csv"
TARGET_COL = "Survived"
TEST_SIZE = 0.25 # 25% of data reserved for testing
RANDOM_SEED = 42 # Random seed for reproducibility
C_REG = 1.0 # Inverse of regularization strength (smalle C = stronger regularization)

# Load Dataset
df = pd.read_csv(CSV_PATH)
print("Data loaded. Shape: ", df.shape)
print("Columns:", df.columns.tolist()[:10], ". . . ")

# Check if target column exists
if TARGET_COL not in df.columns:
    raise ValueError(f"Target Column '{TARGET_COL}' not found in dataset")

y = df[TARGET_COL]  # Variable we want to predict
X = df.drop(columns=[TARGET_COL])  # Feature used for prediction

# Ensure target is binary
unique_y = sorted(pd.Series(y).dropna().unique().tolist())
if len(unique_y) > 2:
    raise ValueError(f"Target column has {len(unique_y)} unique values ({unique_y}); expected binary 0/1")


