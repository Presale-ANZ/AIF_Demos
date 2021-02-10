import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Main(object):
    def __init__(self):

        with open("tokenizer.pickle", "rb") as handle:
            self.tokenizer = pickle.load(handle)

        with open("label_tokenizer.pickle", "rb") as handle:
            self.label_tokenizer = pickle.load(handle)

        model_file = r"./model.h5"
        self.model = tf.keras.models.load_model(model_file)
        self.max_length = 1000

    def predict(self, payload):
        text = [payload]
        seq = self.tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=self.max_length)
        pred = self.model.predict(padded)
        return [
            key
            for key, value in self.label_tokenizer.word_index.items()
            if value == np.argmax(pred)
        ][0]
