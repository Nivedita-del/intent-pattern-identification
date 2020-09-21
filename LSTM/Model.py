import pandas as pd
import numpy as np
import os
import pickle
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.applications import Xception
from keras import regularizers
from keras import backend as K
import keras
import numpy as np
import pandas as pd
import cv2
import os
import glob
import math
import matplotlib.pyplot as plt

seed = 120
np.random.seed(seed)
train_path = 'data/train/train.csv'
train_df = pd.read_csv(train_path)# Loading a csv file with headers
train_df.head()

X_train = train_df["sentence"].fillna("fillna").values
Y_train = train_df[["BookRestaurant", "GetWeather", "PlayMusic", "RateBook"]].values

print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)

print("-------------------------First 2 rows of X_train numpy array--------------------")
for i in range(0, 2):
    print(i, '.', X_train[i])

Tokenizer = Tokenizer()
print(X_train[0])  # training dataset 1st sentence

print("(Input->Sentence) Length of X_train:", X_train.shape)  # Input -> Input
print("(output->Labels) Length of Y_train:", Y_train.shape)  # output -> Labels
texts = X_train
print(texts[0])

Tokenizer.fit_on_texts(texts)
Tokenizer_vocab_size = len(Tokenizer.word_index) + 1
print("Tokenizer vocabulary size:", Tokenizer_vocab_size)

len(max(X_train, key=len))

maxWordCount = 5500
maxDictionary_size = Tokenizer_vocab_size

print("(Input->Sentence) Length of X_train:", X_train.shape)  # Input -> Input
print("(output->Labels) Length of Y_train:", Y_train.shape)  # output -> Labels

num_test_samples = 1586  # Test samples for validation

# Phase 1: Setting up data for training
X_train = X_train[num_test_samples:]  # 1586 samples to n ----> Sentence (Input)
Y_train = Y_train[num_test_samples:]  # 1586 samples to n ----> Labels (Output)

# Phase 2: Setting up data for validation
X_val = X_train[:num_test_samples]  # First 1586 Samples --> Sentence (Input)
Y_val = Y_train[:num_test_samples]  # First 1586 Samples --> Labels (Output)

print("(Input->Sentence) Length of X_train:", X_train.shape)  # Input -> Input
print("(output->Labels) Length of Y_train:", Y_train.shape)  # output -> Labels

# Phase 3: Encoding Operation--->Turn text into a numerical array(using Tokenizer.texts_to_sequences)--->Uses Tokenizer_word_index.
X_train_encoded_words = Tokenizer.texts_to_sequences(X_train)
X_val_encoded_words = Tokenizer.texts_to_sequences(X_val)

print("(output->Labels) Length of Y_train:",
      Y_train.shape)  # output -> Labelsprint("(Input->Sentence) Length of X_train:",len(X_train_encoded_words)) # Input -> Input

X_train_encoded_padded_words = sequence.pad_sequences(X_train_encoded_words, maxlen=maxWordCount)
X_val_encoded_padded_words = sequence.pad_sequences(X_val_encoded_words, maxlen=maxWordCount)

print("(Input->Sentence) Length of X_train:", X_train_encoded_padded_words.shape)  # Input -> Input
print("(output->Labels) Length of Y_train:", Y_train.shape)  # output -> Labels

print(Y_train.shape)
print(Y_val.shape)

# model
model = Sequential()

model.add(Embedding(maxDictionary_size, 32, input_length=maxWordCount))  # to change words to ints
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.5))
# model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# hidden layers
model.add(LSTM(10))
# model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(1200, activation='relu', W_constraint=maxnorm(1)))
# model.add(Dropout(0.6))
model.add(Dense(500, activation='relu', W_constraint=maxnorm(1)))

# model.add(Dropout(0.5))
# output layer
model.add(Dense(4, activation='softmax'))

# Compile model
# adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

model.summary()

learning_rate = 0.0001
epochs = 10
batch_size = 32  # 32
sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])


print(X_train_encoded_padded_words.shape)
print(Y_train.shape)

print(X_val_encoded_padded_words.shape)
print(Y_val.shape)

# model.fit(Train_input,Train_output,epochs,batch_size,verbose,validation_data=(val_input,val_output))
history = model.fit(X_train_encoded_padded_words, Y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_val_encoded_padded_words, Y_val))

print("=============================== Score =========================================")
# Finally calucating the score.
score = model.evaluate(X_val_encoded_padded_words, Y_val, verbose=1)
print('Test accuracy:', score[1], '%')

phrase = "Book The Oriel in Allison for a party of four."
tokens = Tokenizer.texts_to_sequences([phrase])
tokens = pad_sequences(tokens, maxlen=5500)
prediction = model.predict(np.array(tokens))
i, j = np.where(
    prediction == prediction.max())  # calculates the index of the maximum element of the array across all axis
# i->rows, j->columns
i = int(i)
j = int(j)
print(prediction)
total_possible_outcomes = ['1_BookRestaurant', '2_GetWeather', '3_PlayMusic', '4_RateBook']
print("Result:", total_possible_outcomes[j])

# serialize model to JSON
model_json = model.to_json()
with open("LSTM/data/model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("./data/model.h5")
print("Saved model to disk")
with open('LSTM/data/tokenizer.pickle', 'wb') as handle:
    pickle.dump(Tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

