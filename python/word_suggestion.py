#TODO include a space as a word and allow for guessing for part of word
#TODO Have test and validation data as well

from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time
import json, re
from random import randint

# Have to pad inputs if you want to batch your input text
# src: https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
# . Within a single batch, you must have the same number of timesteps since it must be a tensor (this is typically where you see 0-padding). But between batches there is no such restriction. During inference, you can have any length.
class Word_Suggestion:
    def __init__(self, vocab_size, max_sequence_len, word_to_id, id_to_word):
        self.vocab_size = vocab_size
        self.max_sequence_len = min(max_sequence_len, 30) #i.e. num_steps
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.rnn_units = 1024
        self.embedding_dim = 256 
        # Directory where the checkpoints will be saved
        self.checkpoint_dir = '../training_checkpoints'
        # Name of the checkpoint files
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")
        # <Q>SHOULD I SAVE THE WHOLE MODEL?
        self.checkpoint_callback=tf.keras.callbacks.ModelCheckpoint( filepath=self.checkpoint_prefix, save_weights_only=True)
        self.batch_size = 10
        self.skip_steps = 5 # The number of steps to skip before next batch is taken

    # Build the model, this will need to modified
    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_len))
        self.model.add(tf.keras.layers.LSTM(self.rnn_units,  return_sequences=True))
        self.model.add(tf.keras.layers.Dense(self.vocab_size, activation='softmax') )
        # self.model.add(tf.keras.layers.LSTM(self.rnn_units,  return_sequences=True))
        # self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size, activation='softmax')) )
        ## Configures the model for training.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        #return self.model

    def load_and_build_latest_model(self):
        self.build_model()
        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
    
    def train(self, data, use_generator=False, num_epochs=5, has_checkpoint=False, verbose_opt=1):
        if not use_generator:
            #history = self.model.fit(X, y, epochs=num_epochs, verbose=verbose_opt, callbacks=[self.checkpoint_callback])
            pass
        else:
            # examples_per_epoch = len(text)//seq_length
            # steps_per_epoch = examples_per_epoch//BATCH_SIZE
            # print(data)

             #self.max_sequence_len is divided by 2 because that is the approximate mean
             ## This will allow roughly allow for the entire dataset to be trained on the model each epoch
            steps_per_epoch = len(data.split()) // (self.batch_size * (self.max_sequence_len //2)) - (self.skip_steps - self.max_sequence_len)
            if steps_per_epoch <= 0:
                steps_per_epoch = len(data.split()) // (self.batch_size * self.max_sequence_len)
                if steps_per_epoch <= 0:
                    steps_per_epoch = 1
            
            if has_checkpoint:
                self.model.fit_generator(self.train_generator(data),
                 steps_per_epoch=steps_per_epoch,
                  epochs=num_epochs, verbose=verbose_opt,
                   callbacks=[self.checkpoint_callback])
            else:
                self.model.fit_generator(self.train_generator(data),
                 steps_per_epoch=steps_per_epoch,
                  epochs=num_epochs, verbose=verbose_opt)

    def train_generator(self, data):
        """Generates training data for model
        
        # Arguments
            data: A string to be sequenced into chunks for training
            max_sequence_length: A integer representing the maximum sequence length
        # Yields
            A tuple of the training pair
        """
        data_encoded = Suggest_Util.text_to_id(data, self.word_to_id)
        cur_index = 0
        #TODO: Train on variable number of steps
        # x = np.zeros((self.batch_size, self.max_sequence_len ))
        # y = np.zeros((self.batch_size, self.max_sequence_len, self.vocab_size))
        while True:
            batch_sequence_len = randint(1, self.max_sequence_len)
            x = np.zeros((self.batch_size, self.max_sequence_len ))
            y = np.zeros((self.batch_size, self.max_sequence_len, self.vocab_size))
            for i in range(self.batch_size):
                if cur_index + batch_sequence_len >= len(data_encoded):
                    cur_index = 0
                x[i, :] = tf.keras.preprocessing.sequence.pad_sequences([data_encoded[cur_index: cur_index + batch_sequence_len]], maxlen=self.max_sequence_len, padding='pre')
                temp_y = tf.keras.preprocessing.sequence.pad_sequences([data_encoded[cur_index + 1: cur_index + batch_sequence_len + 1]], maxlen=self.max_sequence_len, padding='pre')
                # print("X: ", " ".join([self.id_to_word[str(w)] for w in data_encoded[cur_index: cur_index + self.max_sequence_len]]) )
                # print("Y: ", " ".join([self.id_to_word[str(w)] for w in data_encoded[cur_index + 1: cur_index + self.max_sequence_len  +1]]) ) 
                y[i, :, :] = tf.keras.utils.to_categorical(temp_y, num_classes=self.vocab_size)
                cur_index  += randint(1, self.skip_steps * 2)
            yield x, y

    def predict(self, text, top_num=5):
        """Predict the word that could come next in the given text

        # Arguments
            text: A string to do the prediction on
            top_num: A integer representing the number of top word predictions to send

        # Returns
            A list of the top_num words
        """
        text = Suggest_Util.clean_data(text)
        encoded = Suggest_Util.text_to_id(text, self.word_to_id)
        encoded_padded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=self.max_sequence_len, padding='pre')

        prediction = self.model.predict(encoded_padded)
        prediction = tf.squeeze(prediction, 0)
        predict_words = np.argsort(prediction[self.max_sequence_len-1, :])[::-1][:top_num]
        
        return [self.id_to_word[str(word)] for word in predict_words]

    # generate a sequence from a language model
    def generate_seq(self, text, n_words):
        in_text = Suggest_Util.clean_data(text)
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = Suggest_Util.text_to_id(in_text, self.word_to_id)
            # pre-pad sequences to a fixed length
            encoded_padded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=self.max_sequence_len, padding='pre')
            # predict probabilities for each word
            prediction = self.model.predict(encoded_padded)
            prediction = tf.squeeze(prediction, 0)
            predict_word = tf.multinomial(prediction, num_samples=1)[-1,0].numpy()
            # Get the most likely word
            #predict_word = np.argmax(prediction[:, self.max_sequence_len-1, :])
            # map predicted word index to word
            out_word = self.id_to_word[str(predict_word)]
            # append to input
            in_text += ' ' + out_word
        return in_text
    



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
        uniq_words = set(text.split(" "))
        word_to_id = {word:i for i, word in enumerate(uniq_words)}
        id_to_word = {str(v):k for k,v in word_to_id.items()}
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
    def remove_whitespace(text):
        return " ".join(text.split())


    @staticmethod
    def parse_conversation_json(file_name, max_lines=-1):
        """Parse the json conversation data

        # Arguments
            file_name: A string of the json file name
            max_lines: The max number of text lines to read from 

        # Returns
            A tuple of the dialogue of type string nd the largest word sequence of type integer
        """
        dialogue = ""
        conv = Suggest_Util.load_dict(file_name)
        largest_seq = -1
        i = 0
        for line in conv:
            dialogue += " " + line["text"]
            seq_len = len(line["text"].split(" "))
            if seq_len > largest_seq:
                largest_seq = seq_len
            if max_lines != -1:
                i += 1
                if i >= max_lines:
                    return dialogue, largest_seq
        return dialogue, largest_seq

    @staticmethod
    def remove_nonalphabet_chars(text):
        return re.sub("[^a-zA-Z\'$% ]", "", text)
        
    @staticmethod
    def clean_data(data):
        data = Suggest_Util.remove_whitespace(data.lower())
        return Suggest_Util.remove_nonalphabet_chars(data)
    
    @staticmethod
    def save_dict(metadata, file_name="../config/model_metadata.json"):
        with open(file_name, 'w') as f:
            json.dump(metadata, f)

    @staticmethod
    def load_dict(file_name="../config/model_metadata.json"):
        with open(file_name, 'r') as f:
            return json.load(f)
