#TODO include a space as a word and allow for guessing for part of word

from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time
import json, re

# Have to pad inputs if you want to batch your input text
# src: https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
# . Within a single batch, you must have the same number of timesteps since it must be a tensor (this is typically where you see 0-padding). But between batches there is no such restriction. During inference, you can have any length.
class Word_Suggestion:
    def __init__(self, vocab_size, max_sequence_len=75, word_to_id, id_to_word):
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        # self.path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        self.rnn_units = 1024
        self.embedding_dim = 256 
        # Directory where the checkpoints will be saved
        self.checkpoint_dir = './training_checkpoints'
        # Name of the checkpoint files
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")
        # <Q>SHOULD I SAVE THE WHOLE MODEL?
        self.checkpoint_callback=tf.keras.callbacks.ModelCheckpoint( filepath=self.checkpoint_prefix, save_weights_only=True)

    # Build the model, this will need to modified
    def build_model(self):
        # self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_len-1))
        # self.model.add(tf.keras.layers.LSTM(self.rnn_units))
        # self.model.add(tf.keras.layers.Dense(self.vocab_size, activation='softmax'))
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_len-1))
        self.model.add(tf.keras.layers.LSTM(self.rnn_units))
        self.model.add(tf.keras.layers.Dense(self.vocab_size, activation='softmax'))
        ## Configures the model for training.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #return self.model

    def load_model(self):
        pass
    
    def train(self, X, y, num_epochs=5, verbose_opt=2):
        history = self.model.fit(X, y, epochs=num_epochs, verbose=verbose_opt, callbacks=[self.checkpoint_callback])
    
    # generate a sequence from a language model
    def generate_seq(self, seed_text, n_words):
        in_text = seed_text
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = Suggest_Util.text_to_id(in_text, self.word_to_id)
            # pre-pad sequences to a fixed length
            encoded_padded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=self.max_sequence_len-1, padding='pre')
            # predict probabilities for each word
            yhat = self.model.predict_classes(encoded_padded, verbose=0)[0]
            # map predicted word index to word
            out_word = self.id_to_word[yhat]
            # append to input
            in_text += ' ' + out_word
        return in_text


    #     # <Q> WHAT IS A GOOD EPOCH? WHAT IS A GOOD STEPS_PER_EPOCH 
    #     history = self.model.fit(dataset.repeat(), epochs=self.EPOCHS, steps_per_epoch=self.steps_per_epoch, callbacks=[self.checkpoint_callback])

    # def load_shakespeare(self):
    #     pass

    # def split_input_target(self, chunk):
    #     input_text = chunk[:-1]
    #     target_text = chunk[1:]
    #     return input_text, target_text


class Suggest_Util:
    #TODO include a way to update old one
    #@return (word_to_id, id_to_word)
    @staticmethod
    def words_to_id(text, old_words_to_id=None):
        """Create a mapping between each word and a unique index

        # Arguments
            text: A string to be converted to a list of integers

        # Returns
            A tuple composed of word to id dictionary and id to word dictionary
        """
        text = Suggest_Util.remove_escape_characters(text)
        uniq_words = set(text.split(" "))
        word_to_id = {word:i for i, word in enumerate(uniq_words)}
        id_to_word = {v:k for k,v in word_to_id.items()}
        return word_to_id, id_to_word

    @staticmethod
    def text_to_id(text, word_to_id_dict):
        """Convert all of the text to id by using the word to id dictionary

        # Arguments
            text: A string to be converted to a list of integers
            word_to_id_dict: A dictionary used to get the index of each word

        # Returns
            The index of integer equivalents
        """
        text = Suggest_Util.remove_escape_characters(text)
        return [word_to_id_dict[word] for word in text.split(" ") if word in word_to_id_dict]
        
    @staticmethod
    def remove_escape_characters(text):
        """Remove all escape characters

        # Arguments
            text: A string to be cleansed of escape character

        # Returns
            The string without escape characters and lowercased
        """
        text_removed_escape = list(map(lambda x: x.replace("\\", "").replace("'", "").strip().lower(), re.split(r"(?<=\\)[a-z]{1}", repr(text))))
        text_removed_extra_spaces = list(filter(lambda x: x != "", text_removed_escape))
        return " ".join(text_removed_extra_spaces)

    @staticmethod
    def parse_conversation_json(file_name):
        dialogue = ""
        conv = Suggest_Util.load_dict(file_name)
        for line in conv:
            dialogue += "\n" + line["text"]
        return dialogue

    @staticmethod
    def train_gen(data, max_sequence_length):
        """Generates training data for model
        
        # Arguments
            data: A string to be sequenced into chunks for training
            max_sequence_length: A integer representing the maximum sequence length
        # Yields
            A tuple of the training pair
        """
        for line in data.split('\n'):
            encoded = Suggest_Util.text_to_id(line, word_to_id)
            for i in range(1, len(encoded)):
                sequence = encoded[:i+1]
                sequences_np_arr = np.array(tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_sequence_length, padding='pre'))
                X, temp_y = sequences_np_arr[:,:-1], sequences_np_arr[:,-1]
                y[i,:, :] = tf.keras.utils.to_categorical(y, num_classes=vocab_size)



    @staticmethod
    def save_dict(word_to_id, file_name="word_to_id.json"):
        with open(file_name, 'w') as f:
            json.dump(word_to_id, f)

    @staticmethod
    def load_dict(file_name="word_to_id.json"):
        with open(file_name, 'r') as f:
            return json.load(f)


# text = "QUEENE: I had thought thou hadst a Roman; for the oracle, Thus by All bids the man against the word,Which are so weak of care, by old care done; Your children were in your holy love, And the precipitation through the bleeding throne."

# word_to_id_dict = Suggest_Util.words_to_id(text)
# print(Suggest_Util.text_to_id("QUEENE: I had thought thou", word_to_id_dict ))

data = """Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """

def main():
    # data = """Jack and Jill went up the hill\n
	# 	To fetch a pail of water\n
	# 	Jack fell down and broke his crown\n
	# 	And Jill came tumbling after\n """
    data = Suggest_Util.parse_conversation_json("data/conversation.json")
    # word_to_id, id_to_word = Suggest_Util.words_to_id(data)
    # vocab_size = len(word_to_id)
    
    # sequences = list()
    # max_length = 0
    # for line in data.split('\n'):
    #     encoded = Suggest_Util.text_to_id(line, word_to_id)
    #     for i in range(1, len(encoded)):
    #         sequence = encoded[:i+1]
    #         if len(sequence) > max_length:
    #             max_length = len(sequence)
    #         sequences.append(sequence)
    # sequences_np_arr = np.array(tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='pre'))

    # X, y = sequences_np_arr[:,:-1], sequences_np_arr[:,-1]
    # y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    # word_suggest = Word_Suggestion(vocab_size, max_length, word_to_id, id_to_word)
    # word_suggest.build_model()
    # word_suggest.train(X,y, 10)
    # print(word_suggest.generate_seq("There", 10))
    # word_suggest.model.predict(tf.keras.preprocessing.sequence.pad_sequences(np.array("Jack"), maxlen=max_length, padding='pre'))

    

if __name__ == "__main__":
    main()