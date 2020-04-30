import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, naive_bayes 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
import re


# def create_corpus(tweets):
#     corpus = {}
#     for str in tweets:
#         pat = re.compile(r'[^a-zA-Z ]+')
#         str = re.sub(pat, '', str).lower()

#         #split string
#         splits = str.split()

#         #for loop to iterate over words array
#         for split in splits:
#             if(not split[:4] == "http"):
#                 if(split in corpus):
#                     corpus[split] = corpus[split] + 1
#                 else:
#                     corpus[split] = 1
#     return corpus


train = pd.read_csv("Data/train.csv")
test_vectors = pd.read_csv("Data/test.csv")

# train = train[train['target']==0]
text = train["text"].values
target = train["target"].values

# countwords = len(text)

# corpus = create_corpus(text)

# print(corpus)
        
# top=sorted(corpus.items(), key=lambda x:x[1],reverse=True)[:10] 

# x,y=zip(*top)
# plt.bar(x,y)
# plt.title("Target: 0, No. of tweets:{}".format(countwords))
# plt.show()

text_train, text_test, y_train, y_test = train_test_split(text,target,test_size=0.25,random_state = 42)

vectorizer = TfidfVectorizer()
vectorizer.fit(text_train)

x_train = vectorizer.transform(text_train)
x_test  = vectorizer.transform(text_test)
x_train

test_vectors = vectorizer.transform(test_vectors["text"])

input_dim = x_train.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=30, verbose=False, validation_data=(x_test, y_test), batch_size=10)

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

submission = pd.read_csv("sample_submission.csv")

predictions = np.around(model.predict(test_vectors))
submission['target'] = predictions.astype(int) 
submission.to_csv('submission.csv', index=False)