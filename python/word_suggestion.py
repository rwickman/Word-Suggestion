#TODO allow for guessing for part of word
#TODO Have test and validation data as well

from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from random import randint
import os, time, json, re
from math import ceil


class Word_Suggestion:
    def __init__(self, vocab_size, max_sequence_len, word_to_id, id_to_word, checkpoint_dir='../models/training_checkpoints/conversation'):
        self.vocab_size = vocab_size
        self.input_length = min(max_sequence_len, 40) #i.e. num_steps
        self.max_sequence_length = max_sequence_len
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.rnn_units = 1024
        self.embedding_dim = 256 
        # Directory where the checkpoints will be saved
        self.checkpoint_dir = checkpoint_dir
        # Name of the checkpoint files
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")
        self.checkpoint_callback=tf.keras.callbacks.ModelCheckpoint( filepath=self.checkpoint_prefix, save_weights_only=True)
        self.batch_size = 10
        self.skip_steps = 5 # The number of steps to skip before next batch is taken
        self.clean_conv_data = None

    # Build the model, this will need to modified
    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.input_length))
        self.model.add(tf.keras.layers.LSTM(self.rnn_units,  return_sequences=True))
        self.model.add(tf.keras.layers.Dense(self.vocab_size, activation='softmax') )
        # self.model.add(tf.keras.layers.LSTM(self.rnn_units,  return_sequences=True))
        # self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size, activation='softmax')) )
        ## Configures the model for training.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def load_and_build_latest_model(self):
        self.build_model()
        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
    
    def train(self, data, generator="", num_epochs=5, has_checkpoint=False, verbose_opt=1):
        if not generator:
            X = data[0]
            y = data[1]
            history = self.model.fit(X, y, epochs=num_epochs, verbose=verbose_opt, callbacks=[self.checkpoint_callback])
        elif generator == "conversation":
            # examples_per_epoch = len(text)//seq_length
            # steps_per_epoch = examples_per_epoch//BATCH_SIZE
            # print(data)

             #self.input_length is divided by 2 because that is the approximate mean
             ## This will allow roughly allow for the entire dataset to be trained on the model each epoch
            steps_per_epoch = len(data.split()) // (self.batch_size * (self.input_length //2)) - (self.skip_steps - self.input_length)
            if steps_per_epoch <= 0:
                steps_per_epoch = len(data.split()) // (self.batch_size * self.input_length)
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

        elif generator == "lyrics":
            steps_per_epoch =  ceil( (len(data)  * self.max_sequence_length - 1 )/ self.batch_size)
            if has_checkpoint:
                self.model.fit_generator(self.train_generator_lyrics(data),
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs, verbose=verbose_opt,
                callbacks=[self.checkpoint_callback])
            else:
                self.model.fit_generator(self.train_generator_lyrics(data),
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs, verbose=verbose_opt)
        else:
            raise Exception("INVALID GENERATOR")
    
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
        while True:
            batch_sequence_len = randint(1, self.input_length + 1)
            x = np.zeros((self.batch_size, self.input_length ))
            y = np.zeros((self.batch_size, self.input_length, self.vocab_size))
            for i in range(self.batch_size):
                if cur_index + batch_sequence_len >= len(data_encoded):
                    cur_index = 0
                x[i, :] = tf.keras.preprocessing.sequence.pad_sequences([data_encoded[cur_index: cur_index + batch_sequence_len]], maxlen=self.input_length, padding='pre')
                temp_y = tf.keras.preprocessing.sequence.pad_sequences([data_encoded[cur_index + 1: cur_index + batch_sequence_len + 1]], maxlen=self.input_length, padding='pre')
                # print("X: ", " ".join([self.id_to_word[str(w)] for w in data_encoded[cur_index: cur_index + self.input_length]]) )
                # print("Y: ", " ".join([self.id_to_word[str(w)] for w in data_encoded[cur_index + 1: cur_index + self.input_length  +1]]) ) 
                y[i, :, :] = tf.keras.utils.to_categorical(temp_y, num_classes=self.vocab_size)
                cur_index  += randint(1, self.skip_steps * 2)
            yield x, y
    
    def train_generator_lyrics(self, data):
        while True:
            for line in data:
                seq = []
                encoded_line = Suggest_Util.text_to_id(line, self.word_to_id)
                for i in range(1, len(encoded_line)):
                    seq.append(encoded_line[:i+1])
                    if (i+1) % self.batch_size == 0:
                        #Padding will  either add 0's or remove the first few elements depending on the size
                        sequences = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.input_length + 1, padding='pre')
                        seq = []
                        sequences = np.array(sequences)
                        X, y = sequences[:,:-1], sequences[:,1:]
                        y =  tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)
                        yield X, y

    def train_update(self, data):
        try:
            #Consider spanwing this off in a different thread
            
            self.word_to_id, self.id_to_word = Suggest_Util.words_to_id(data, True, self.word_to_id)
            self.vocab_size = 10000

            if not self.clean_conv_data:
                conv_data, max_sequence_length = Suggest_Util.parse_conversation_json("../data/conversation.json", 25)
                self.clean_conv_data = Suggest_Util.clean_data(conv_data)
            
            for item in data:
                for word in item.split():
                    self.clean_conv_data += " " + word
            
            self.train(self.clean_conv_data,"conversation", 1, False)
        
            model_metadata = {"vocab_size" : self.vocab_size, "max_sequence_length" : self.max_sequence_length, "word_to_id" : self.word_to_id, "id_to_word" : self.id_to_word, "checkpoint_dir" : '../models/training_checkpoints/conversation'}
            Suggest_Util.save_dict(model_metadata, "../config/updated_model_metadata.json")
            
            self.model.save('../models/training_checkpoints/conversation/conv_model.h5')
        except Exception as e:
            print("ERROR DURING train_update: ", e, e.__traceback__)

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
        encoded_padded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=self.input_length, padding='pre')

        prediction = self.model.predict(encoded_padded)
        prediction = tf.squeeze(prediction, 0)
        predict_words = np.argsort(prediction[self.input_length-1, :])[::-1][:top_num]
        
        return [self.id_to_word[str(word)] for word in predict_words]

    # generate a sequence from a language model
    def generate_seq(self, text, n_words):
        in_text = Suggest_Util.clean_data(text)
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = Suggest_Util.text_to_id(in_text, self.word_to_id)
            # pre-pad sequences to a fixed length
            encoded_padded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=self.input_length, padding='pre')
            # predict probabilities for each word
            prediction = self.model.predict(encoded_padded)
            prediction = tf.squeeze(prediction, 0)
            predict_word = tf.multinomial(prediction, num_samples=1)[-1,0].numpy()
            # Get the most likely word
            #predict_word = np.argmax(prediction[:, self.input_length-1, :])
            # map predicted word index to word
            out_word = self.id_to_word[str(predict_word)]
            # append to input
            in_text += ' ' + out_word
        return in_text



class Suggest_Util:
    #TODO include a way to update old one
    #@return (word_to_id, id_to_word)
    @staticmethod
    def words_to_id(text, is_list=False, old_word_to_id=None):
        """Create a mapping between each word and a unique index

        # Arguments
            text: A string or list to be converted to a list of integers
            is_list: True if text is a list
            old_words_to_id: A dictionary of the words_to_id that need to be updated

        # Returns
            A tuple composed of word to id dictionary and id to word dictionary
        """
        if is_list:
            x = ""
            for line in text:
                x += line + " "
            text = x
        
        uniq_words = set(text.split(" "))
        
        if old_word_to_id:
            word_to_id = old_word_to_id
            start = len(old_word_to_id)
            for word in uniq_words:
                if word not in word_to_id:
                    word_to_id[word] = start
                    start += 1
        else:
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
            A list of the id equivalents
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
    def parse_movie_quotes(file_name, name="quote"):
        quotes = []
        movie_quotes = Suggest_Util.load_dict(file_name)
        largest_seq = -1
        for line in movie_quotes:
            quotes.append(Suggest_Util.clean_data(line[name]))
            seq_len = len(line[name].split(" "))
            if seq_len > largest_seq:
                largest_seq = seq_len
        return quotes, largest_seq

    @staticmethod
    def split_by_word_sequentially(data, word_to_id_dict, max_sequence_len):
        """Create the training dataset by splitting all lines by word and after it (this is hard to explain)
        EX: data = ["how are you"]
        X = ["how", "how are"]
        y = ["are", "you"]

        # Arguments
            data: A list of strings
            word_to_id_dict: A dictionary used to get the index of each word 

        # Returns
            A tuple of X and y training data
        """
        seq = []
        for line in data:
            encoded_line = Suggest_Util.text_to_id(line, word_to_id_dict)
            for i in range(1, len(encoded_line)):
                seq.append(encoded_line[:i+1])
        sequences = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_sequence_len, padding='pre')
        sequences = np.array(sequences)
        X, y = sequences[:,:-1], sequences[:,1:]
        y =  tf.keras.utils.to_categorical(y, num_classes=len(word_to_id_dict))
        return X, y        
            
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
