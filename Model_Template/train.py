import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from utils import Util


class Main(object):
    def __init__(self):
        self.vocab_size = 10000  # make the top list of words (common words)
        self.embedding_dim = 64
        self.max_length = 1000
        self.trunc_type = "post"
        self.padding_type = "post"
        self.oov_tok = "<OOV>"  # OOV = Out of Vocabulary
        self.num_epochs = 5
        self.model = tf.keras.Model()
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_tok)
        self.label_tokenizer = Tokenizer(filters="", split="$")

    def train(self, directory):
        train_email_bodies, train_labels = Util.read_csv(directory + "/train.csv")

        self.tokenizer.fit_on_texts(train_email_bodies)
        word_index = self.tokenizer.word_index

        train_sequences = self.tokenizer.texts_to_sequences(train_email_bodies)
        train_padded = pad_sequences(
            train_sequences,
            maxlen=self.max_length,
            padding=self.padding_type,
            truncating=self.trunc_type,
        )

        self.label_tokenizer.fit_on_texts(train_labels)

        training_label_seq = np.array(
            self.label_tokenizer.texts_to_sequences(train_labels)
        )
        # label_tokenizer.word_index
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(self.vocab_size, 64),
                tf.keras.layers.Conv1D(128, 5, activation="relu"),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(12, activation="sigmoid"),
            ]
        )

        opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )

        history = self.model.fit(
            train_padded, training_label_seq, epochs=self.num_epochs, verbose=2
        )

        return self.model

    def save(self):
        with open("tokenizer.pickle", "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("label_tokenizer.pickle", "wb") as handle:
            pickle.dump(self.label_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.model.save("model.h5")
