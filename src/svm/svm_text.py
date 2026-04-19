# ============================================================
# 1. IMPORTS
# ============================================================
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

# ============================================================
# 2. LOAD DATA
# ============================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_FILE = os.path.join(DATA_DIR, "spam.csv")

df = pd.read_csv(INPUT_FILE, encoding='latin-1')

# Adjust depending on dataset
df = df.rename(columns={"v1": "label", "v2": "message"})

# Convert target
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

# ============================================================
# 3. TEXT VECTORIZATION (TF-IDF)
# ============================================================
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 5. MODELS (SVM)
# ============================================================
models = {
    "Linear": SVC(kernel='linear'),
    "RBF": SVC(kernel='rbf'),
    "Polynomial": SVC(kernel='poly', degree=2)
}

results = {}

# ============================================================
# 6. TRAIN + EVALUATE
# ============================================================
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=2)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "cv": cv_scores.mean(),
        "acc": acc,
        "f1": f1,
        "cm": cm
    }

# ============================================================
# 7. PRINT RESULTS
# ============================================================
for name, res in results.items():
    print(f"\n=== {name} SVM ===")
    print("CV:", res["cv"])
    print("Accuracy:", res["acc"])
    print("F1:", res["f1"])

best_model_name = max(results, key=lambda x: results[x]["f1"])
print(f"\nBest Model: {best_model_name}")
# ============================================================
# 8. PLOT CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
class_names = [ 'Ham','Spam']

for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=res["cm"],display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(name)

plt.tight_layout()
plt.show()