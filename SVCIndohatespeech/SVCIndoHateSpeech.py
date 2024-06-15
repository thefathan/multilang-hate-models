import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Load the dataset
df = pd.read_csv('/nas.dbms/fathan/test/data_preprocessed.csv', header=0)
df.info()
print(df.sample(5))

# Machine Learning
df = df.dropna()
X = df['text']
y = df['hs_class']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the parameter grid for GridSearchCV
param_grid = {
    'clf__C': [0.1, 0.5, 1, 10, 100],
    'clf__penalty': ['l2'],
    'clf__loss': ['squared_hinge'],
    'clf__dual': [True],
    'clf__tol': [0.0001, 0.001],
    'clf__max_iter': [1000, 5000],
    'tfidf__lowercase': [True, False],
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
    'tfidf__stop_words': ['english', ENGLISH_STOP_WORDS, None]
}

# Create a pipeline
txt_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Perform GridSearchCV
search = GridSearchCV(txt_clf, param_grid, refit=True)
search.fit(X_train, y_train)

# Make predictions
predictions = search.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions, pos_label='positive')
rec = recall_score(y_test, predictions, pos_label='positive')
f1 = f1_score(y_test, predictions, pos_label='positive')

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)

# Compute confusion matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('/nas.dbms/fathan/test/SVCIndohatespeech/confusion_matrix.png')  # Save the plot as a PNG file
print("Confusion matrix saved as 'confusion_matrix.png'")
