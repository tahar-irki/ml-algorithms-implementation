import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

# ============================================================
# 2. LOAD DATA
# ============================================================
# Project structure navigation
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "Teen_Mental_Health_Dataset.csv")

# Load dataset
df = pd.read_csv(INPUT_FILE)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
# Define numerical and categorical features
num_features = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level"
]

cat_features = [
    "gender",
    "platform_usage",
    "social_interaction_level"
]

# Target variable
target = "depression_label"

# Split features and target
X = df[num_features + cat_features]
y = df[target]

# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================
# - Standardize numerical features
# - One-hot encode categorical features

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================
# We evaluate three SVM kernels
kernels = {
    "Linear": "linear",
    "Polynomial": "poly",
    "RBF": "rbf"
}

results = {}

# ============================================================
# 7. TRAINING + CROSS-VALIDATION + EVALUATION
# ============================================================
for name, kernel in kernels.items():
    
    # Full pipeline: preprocessing + model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("svc", SVC(kernel=kernel))
    ])

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    # Store results
    results[name] = {
        "model": model,
        "cv_mean": cv_scores.mean(),
        "accuracy": acc,
        "f1": f1,
        "cm": cm
    }

# ============================================================
# 8. RESULTS SUMMARY
# ============================================================
for name, res in results.items():
    print(f"\n=== {name} SVM ===")
    print(f"Cross-Validation Accuracy: {res['cv_mean']:.4f}")
    print(f"Test Accuracy: {res['accuracy']:.4f}")
    print(f"F1 Score: {res['f1']:.4f}")

# Best model selection based on F1 score
best_model_name = max(results, key=lambda x: results[x]["f1"])
print(f"\nBest Model: {best_model_name}")

# ============================================================
# 9. CONFUSION MATRIX VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
class_names = ['Not depressed', 'Depressed']
for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=res["cm"],display_labels=class_names)
    disp.plot(ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.show()

# ============================================================
# 10. PIPELINE DIAGRAM (FOR REPORT)
# ============================================================
# This section prints a simple textual diagram you can include
# in your report or convert into a figure.



# ============================================================
# END OF SCRIPT
# ============================================================
