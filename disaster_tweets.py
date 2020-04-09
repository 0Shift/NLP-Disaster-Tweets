import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


train_df = pd.read_csv("Data/train.csv")
test_df = pd.read_csv("Data/test.csv")



count_vectorizer = feature_extraction.text.CountVectorizer()


example_train_vectors = count_vectorizer.fit_transform(train_df["text"])

train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])

print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())

model = Sequential()

model.add(LSTM(128, activation='relu', return_sequences = True))
#model.add(Dropout(0.1))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

model.fit(train_vectors, epochs = 3, validation_data = test_vectors)