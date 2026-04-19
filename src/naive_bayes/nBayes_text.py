import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#loading the data from my data folder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "spam.csv")

try:
    df = pd.read_csv(INPUT_FILE, encoding='latin-1')
    df.columns = ['type', 'text'] + list(df.columns[2:])
except FileNotFoundError:
    print("Dataset not found")
    exit()

#Term Frequency-Inverse Document Frequency
#TF(t,d)=count of t in d / total words in d​
#IDF(t)=log( 1+N / 1+DF(t) ​)+1
# N = total number of documents
# DF(t) = number of documents containing word t
#english keyword drops common words like a, is, are etc ...

vectorizer = TfidfVectorizer(
    stop_words='english',
 
)
#saving the file of bag of words 
#fit_transform transform words to numbers and uses formula TF-IDF to calculate Fq

X_full = vectorizer.fit_transform(df['text'])

#making it human readable

subset_size = 300
full_df = pd.DataFrame(
    X_full[:subset_size].toarray(),
    columns=vectorizer.get_feature_names_out()
)

full_df.index = [f'email{i+1}' for i in range(len(full_df))]
full_df.insert(0, 'Class', df['type'][:subset_size].values)

FULL_PATH = os.path.join(DATA_DIR, "full_word_table.csv")
full_df.to_csv(FULL_PATH)
print(f"Saved count table (subset) to: {FULL_PATH}")

#here start to encode the dataset

df['label'] = df['type'].map({'ham': 0, 'spam': 1})

#now spliting the dataset

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

#vectorizing 
#fit_transform transform words to numbers

X_train_vec = vectorizer.fit_transform(X_train)

#transform drops words that are not in the training set
X_test_vec = vectorizer.transform(X_test)

#the multinomial naive bayes 
# alpha 1.0 is laplace smoothing

model = MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)

#evuluation of my model

y_pred = model.predict(X_test_vec)

#accuracy 

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#matrix 

cm = confusion_matrix(y_test, y_pred)

labels = ["Ham", "Spam"]

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0,1], labels)
plt.yticks([0,1], labels)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#specificity

tn = cm[0,0]
fp = cm[0,1]
specificity = tn / (tn + fp) if (tn + fp) else 0

print("Specificity:", specificity)
