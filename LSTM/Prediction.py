from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
# load json and create model
json_file = open('LSTM/data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./data/model.h5")
print("Loaded model from disk")

# loading
with open('LSTM/data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

phrase = "Will it rain today?."
tokens = tokenizer.texts_to_sequences([phrase])
tokens = pad_sequences(tokens, maxlen=5500)
prediction = loaded_model.predict(np.array(tokens))
i,j = np.where(prediction == prediction.max()) #calculates the index of the maximum element of the array across all axis
# i->rows, j->columns
i = int(i)
j = int(j)
print(prediction)
total_possible_outcomes = ['1_BookRestaurant','2_GetWeather','3_PlayMusic','4_RateBook']
print("Result:",total_possible_outcomes[j])
prediction = prediction.tolist()
prediction = prediction[0]
print(prediction)
list1 = prediction
list2 = total_possible_outcomes
list1, list2 = zip(*sorted(zip(list1, list2), reverse=True))

if __name__ == '__main__':
    list1
    list2

