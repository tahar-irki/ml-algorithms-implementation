# 🚀 Multi-Algorithm Machine Learning Sandbox

A structured Machine Learning project implementing algorithms from **two complementary perspectives**:

- 🧮 **From-Scratch Mathematical Implementations**
- ⚙️ **Production-Ready Library Pipelines**

This repository explores real-world datasets to bridge the gap between:

- 📐 Mathematical theory  
- 🧠 Algorithmic reasoning  
- 🏗 Practical ML engineering  

---

# 📂 Project Structure

```
├── data/                         # Datasets 
│ 
│
├── src/                          # Algorithm implementations
│   │
│   ├── data_loader/              # data loader 
│   │   └── data_loader.py        # data loader (for all the algorithms)
│   │
│   ├── knn/                      # K-Nearest Neighbors
│   │   ├── knnAlgo.py            # Custom KNN (from scratch)
│   │   └── knnLib.py             # scikit-learn benchmark version
│   │
│   ├── naive bayes/              # Naive Bayes classifiers
│   │   ├── nBayes_tabular.py     # Custom Mixed Naive Bayes
│   │   └── nBayes_text.py        # Production pipeline version
│   │
│   ├── svm/                      # Support Vector Machine
│   │   └── svm_tabular_S.py        # tabular version
│   │
│   └── next algorithm/           # Future implementations
│       └── dataFetcher.py
│
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

# 🧠 Core Implementations

## 1️⃣ K-Nearest Neighbors (KNN)

### 🔹 From-Scratch Implementation

A manual implementation of the KNN algorithm including:

- Custom distance computation
- Hybrid metric design:
  - Manhattan distance → Numeric features
  - Hamming distance (0/1) → Categorical features
- Majority voting classification logic

This version demonstrates deep understanding of:

- Distance-based learning  
- Feature engineering  
- Algorithmic complexity  

---

### 🔹 Library-Based Benchmark

Implemented using **scikit-learn’s KNeighborsClassifier** for performance comparison against the custom logic.

This allows benchmarking between:

- Mathematical implementation  
- Optimized production model  

---

### 🎯 Domain: Sports Analytics

The dataset focuses on predicting football player positions:

- DF → Defender  
- MF → Midfielder  
- FW → Forward  
- GK → Goalkeeper  

Predictions are based on real performance statistics.

---

## 2️⃣ Naive Bayes

### 🔹 Custom Mixed Naive Bayes

The implementation supporting mixed feature types:

- Gaussian probability distributions → Continuous academic features  
- Laplace-smoothed categorical probabilities → Demographic features  

This implementation demonstrates understanding of:

- Bayes Theorem  
- Conditional independence assumption  
- Likelihood estimation  
- Numerical stability  

---

### 🔹 Production Pipeline Version

Built using scikit-learn with:

- Pipeline  
- ColumnTransformer  
- SimpleImputer  
- StandardScaler  
- OneHotEncoder  

Capabilities include:

- Automatic missing value handling  
- Feature scaling  
- Categorical encoding  
- Data leakage prevention  
- Clean train/test separation  

---

### 🎯 Domain: Educational Data

The model predicts student dropout risk using:

- Academic performance metrics  
- Socio-demographic variables  

---

## 3️⃣ Support Vector Machines (SVM)

### 🔹 Kernelized Classification Logic

The implementation evaluates multiple geometric approaches to boundary separation:

- Linear Kernel → Finds the optimal hyperplane for linearly separable data.
- RBF (Radial Basis Function) Kernel → Projects features into higher-dimensional space to handle non-linear relationships.

This implementation demonstrates understanding of:

- Max-Margin Hyperplanes (maximizing the distance between classes).
- Support Vectors (identifying critical data points that define the boundary).
- Kernel Trick (computational efficiency in high-dimensional mapping).
- Regularization & Soft Margins (balancing misclassification vs. boundary width).

---

### 🔹 Production Pipeline Version

Built using scikit-learn to ensure a modular and reproducible workflow:

- ColumnTransformer → Parallel processing of disparate data types.
- StandardScaler → Crucial for SVM, as it is sensitive to the scale of input features (Distance-based).
- OneHotEncoder → Transforms categorical identifiers into a sparse numerical format.
- Pipeline → Chains preprocessing and SVC estimator to prevent data leakage during Cross-Validation.

Capabilities include:

- K-Fold Cross-Validation → Robust performance estimation on the training set.
- Automated Preprocessing → Consistent transformation for both training and unseen test data.
- Multi-Metric Evaluation → Comparison via Accuracy, F1-Score, and Confusion Matrices.

---

### 🎯 Domain: Educational Data

The model predicts teen depression risk using high-dimensional behavioral data:

- Daily Habits (Social media hours, sleep duration, screen time).
- Academic & Physical Metrics (Performance scores, activity levels).
- Psychological Indicators (Stress, anxiety, and addiction levels).

---

# 📊 Model Evaluation Strategy

Each algorithm (custom and library-based) is evaluated using:

### ✔ Accuracy  
Overall prediction correctness.

### ✔ Average Specificity  
Measures the ability to correctly identify negative classes across multiple labels.

### ✔ Classification Report  
Includes:
- Precision  
- Recall  
- F1-Score  
- Support  

### ✔ Confusion Matrix  
Visualized using matplotlib to analyze misclassification patterns.

This ensures consistent and fair comparison between implementations.

---

# 🛠 Setup & Usage

## 1️⃣ Installation

```bash
git clone https://github.com/tahar-irki/ml-algorithms-implementation.git
cd ml-algorithms-implementation
pip install -r requirements.txt
```

---

## 2️⃣ Running Algorithms

### Run Custom KNN
```bash
python src/knn/knnAlgo.py
```

### Run Naive Bayes students dropouts
```bash
python "src/naive bayes/nBayesLib.py"
```

### Run Naive Bayes spam/ham
```bash
python "src/naive bayes/nbDataFbagOwords.py"
```

### Run svm tabular
```bash
python src/svm/svm_tabular_S.py
```


---



# 👤 Author

**tahar irki**  
