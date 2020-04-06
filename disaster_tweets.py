import numpy as np 
import pandas as pd 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


# Starter code from: https://www.kaggle.com/philculliton/nlp-getting-started-tutorial

train_df = pd.read_csv("Data/train.csv")
test_df = pd.read_csv("Data/test.csv")


count_vectorizer = feature_extraction.text.CountVectorizer()

## let's get counts for the first 5 tweets in the data
train_vectors = count_vectorizer.fit_transform(train_df["text"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])



## Our vectors are really big, so we want to push our model's weights
## toward 0 without completely discounting different words - ridge regression 
## is a good way to do this.
clf = linear_model.RidgeClassifier()

clf.fit(train_vectors, train_df["target"])

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], scoring="f1")
print(scores)

# sample_submission = pd.read_csv("sample_submission.csv")

# sample_submission["target"] = clf.predict(test_vectors)
# print(sample_submission.head())
