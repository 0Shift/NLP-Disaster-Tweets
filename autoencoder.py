import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, naive_bayes 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
from keras.models import Sequential, LSTM, TimeDistributed, RepeatVector, Dense
from keras import layers
import matplotlib.pyplot as plt
import re


train = pd.read_csv("Data/train.csv")
test_vectors = pd.read_csv("Data/test.csv")

text = train["text"].values
target = train["target"].values


text_train, text_test, y_train, y_test = train_test_split(text,target,test_size=0.25,random_state = 42)

vectorizer = TfidfVectorizer()
vectorizer.fit(text_train)

x_train = vectorizer.transform(text_train)
x_test  = vectorizer.transform(text_test)
x_train

test_vectors = vectorizer.transform(test_vectors["text"])

input_dim = x_train.shape[1]
# model = Sequential()
# model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

# model.fit(x_train, y_train, epochs=300, verbose=False, validation_data=(x_test, y_test), batch_size=10)

n_in = len(x_train)
x_train = x_train.reshape((1, n_in, 1))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')


loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

submission = pd.read_csv("sample_submission.csv")

predictions = np.around(model.predict(test_vectors))
submission['target'] = predictions.astype(int) 
submission.to_csv('submission.csv', index=False)