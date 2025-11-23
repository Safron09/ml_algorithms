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
from sklearn.impute import SimpleImputer    # handle NaNs
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

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y  # Kee class ration consistent across splits

)

print(f"Train size {X_train.shape}, Test Size: {X_test.shape}")


# Preprocessing setup
# Identify numerical and categorical columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()   #float/int columns
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]  # non-numerical columns

# StandardScaler standardize numeric columns to mean=0, std=1
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  
    ("scaler",  StandardScaler())
])  # Helps Logistic regression converge faster and handle varying scales

# OnehotEncoder converts categorical variables into binary vectors
categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine both preprocessings steps using ColumnTransformer
prerpocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, numeric_cols),
        ("cat", categorical_tf, categorical_cols)
    ],
    remainder="drop"
)

# Model Defenition
# Logistic regression is a linear model
# Regularization helps to avoid overfitting
logreg = LogisticRegression(
    penalty = "l2",   # Default; adds L2 penalty (sun of scuares of weights)
    C=C_REG,            # Inverse of regularization strength (1/lambda)
    solver = "liblinear",
    random_state=RANDOM_SEED
)

# Build Full Pipeline

# The Pipeline runs preprocessing and model training together
# Ensure consistent transformation for both
pipe = Pipeline(steps=[
    ("preprocess", prerpocess),
    ("model", logreg)
])

# Train Model

pipe.fit(X_train, y_train)  # Fits both preprocessing and model on training data

# Predict and evaluate

# predict_proba() gives probabilities; [:1] means probability of class 1
y_proba = pipe.predict_proba(X_test)[:,1]
# Converts probabilities into binary predictions using a 0.5 treshold
y_pred = (y_proba >= 0.5).astype(int)


# Compute metrics
acc = accuracy_score(y_test, y_pred)   # Fraction of correct predictions
prec = precision_score(y_test, y_pred)  # TP / (TP + FP)
rec = recall_score(y_test, y_pred)   # TP / (TP + FN)
f1 = f1_score(y_test, y_pred)       # Harmonic mean of precision and recall
auc = roc_auc_score(y_test, y_pred)  # Area under ROC curve (probability ranking quiality)

# print Metrics
print("\n=== Evaluation Metrics ===")
print(f"Accuracy   : {acc:.3f}")
print(f"Precision  : {prec:.3f}")
print(f"Recall     : {rec:.3f}")
print(f"F1 Score   : {f1:.3f}")
print(f"ROC-AUC    : {auc:.3f}")

# Print detailed class-wise report.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# Visualisation folder set up
os.makedirs("ml_algorithms/logistic_regression/plots", exist_ok=True)
PLOT_DIR = "ml_algorithms/logistic_regression/plots"


# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)  # fpr: false positive rate; tpr: true positive rate.
plt.figure()
plt.plot(fpr, tpr, label = f"ROC Curve (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color = 'gray')
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=160)
plt.close()
# the Roc curve shows how recall (TPR) trades off

#  Plot Precision-Recall Curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
# precision_recall_curve() computes precision/recall for different thresholds.
plt.figure()
plt.plot(rec_curve, prec_curve, color="purple")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "precision_recall_curve.png"), dpi=160)
plt.close()
# This helps when dealing with imbalanced datasets (focuses on positive class performance).

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")  # Visualize confusion matrix as an image.
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

# Add numerical labels to cells 
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")

plt.colorbar(im)
plt. tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=160)
plt.close()

# Feature Importance
# Helper function to recover all features names after transformation
def get_feature_names(preprocess, numeric_cols, categorical_cols):
    names = []
    names.extend(numeric_cols)
    if categorical_cols:
        ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_cols)
        names.extend(cat_names.tolist())
    return names

# Get Final features names and their learned weights
feature_names = get_feature_names(pipe.named_steps["preprocess"], numeric_cols, categorical_cols)
coefs = pipe.named_steps["model"].coef_.ravel()  # Coefficients of logistic regression

# Sort by absolute value
k = min(20, len(coefs))
top_idx = np.argsort(np.abs(coefs))[::-1][:k]
top_names = [feature_names[i] for i in top_idx]
top_vals = coefs[top_idx]

plt.figure(figsize=(8, max(4, k * 0.3)))
y_pos = np.arange(k)
plt.barh(y_pos, top_vals, color="teal")
plt.yticks(y_pos, top_names)
plt.gca().invert_yaxis()  # Highest at top.
plt.xlabel("Coefficient Weight")
plt.title("Top Logistic Regression Coefficients (|weight|)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "top_coefficients.png"), dpi=160)
plt.close()
# Positive coefficients increase the chance of class=1; negative decrease it.

print(f"\n📊 Saved plots to: {PLOT_DIR}")
print("Files: roc_curve.png, precision_recall_curve.png, confusion_matrix.png, top_coefficients.png")

    
