import numpy as np 
import pandas as pd 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Starter code from: https://www.kaggle.com/philculliton/nlp-getting-started-tutorial
train_df = pd.read_csv("Data/train.csv")
test_df = pd.read_csv("Data/test.csv")


x_train, x_test, y_train, y_test = train_test_split(train_df["text"],
 train_df["target"], 
 stratify=train_df["target"], 
 random_state=42, 
 test_size=0.10, shuffle=True)



count_vectorizer = feature_extraction.text.CountVectorizer()

count_vectorizer.fit(list(x_train))
train_vectors = count_vectorizer.transform(x_train)
test_vectors = count_vectorizer.transform(x_test)

## Our vectors are really big, so we want to push our model's weights
## toward 0 without completely discounting different words - ridge regression 
## is a good way to do this.
clf = linear_model.RidgeClassifier()
clf.fit(train_vectors, y_train)
scores = model_selection.cross_val_score(clf, test_vectors, y_test, scoring="f1")
print("Rigid Classifier CV F1 Scores:", scores * 100)

tfid_vectorizer = feature_extraction.text.TfidfVectorizer()
tfid_vectorizer.fit(list(x_train))
train_vectors = tfid_vectorizer.transform(x_train)
test_vectors = tfid_vectorizer.transform(x_test)

clf = linear_model.RidgeClassifier()
clf.fit(train_vectors, y_train)
scores = model_selection.cross_val_score(clf, test_vectors, y_test, scoring="f1")
print("Ridge Classifier TFID F1 Scores:", scores * 100)

# sample_submission = pd.read_csv("sample_submission.csv")

# sample_submission["target"] = clf.predict(test_vectors)
# print(sample_submission.head())
