import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GlobalMaxPool1D, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Input
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import os
import random
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib import rcParams
import nltk


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.isri import ISRIStemmer
from collections import Counter
import itertools
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from joblib import dump, load
from time import time
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import os
import pickle

w2v_root = "../Datasets/w2v_model/glove_twitter_200d.model"
data_root = '../Datasets/C/english/data_noaug.p'
label_root = '../Datasets/C/english/label_noaug.p'
file_path = './model/weights.best-lstm-c.hdf5'

def data_load(data_root,label_root):

    data = pickle.load(open(data_root, "rb"))
    label = pickle.load(open(label_root, "rb"))
    return data,label



def load_w2v_model(root):
    w2v_model = KeyedVectors.load(root)
    return w2v_model


def pretrained_layer(embedding_model):
    word_dict = {}
    for word in list(embedding_model.index_to_key):
        word_dict[word] = embedding_model[word]

    Embedding_dim = embedding_model.vector_size

    word2idx = {'PAD': 0}

    vocab_list = [word for word in enumerate(embedding_model.key_to_index.keys())]
    embeddings_matrix = np.zeros((len(vocab_list) + 1, embedding_model.vector_size))

    for i in range(len(vocab_list)):
        word = vocab_list[i][1]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = word_dict[word]


    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                output_dim=Embedding_dim,
                                weights=[embeddings_matrix],
                                trainable=False)
    return embedding_layer




def create_model(tweet,topic,embedding_layer):
    inputA = Input(shape=tweet[0].shape)
    inputB = Input(shape=topic[0].shape)

    # the first branch operates on the first input
    x = embedding_layer(inputA)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128,return_sequences=True), merge_mode = 'concat')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.4)(x)
    x = Model(inputs=inputA, outputs=x)

    # the second branch opreates on the second input
    y = embedding_layer(inputB)
    t = Dropout(0.3)(y)
    #y = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(y)
    #y = MaxPooling1D(pool_size=2)(y)
    #y = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(y)
    #y = MaxPooling1D(pool_size=2)(y)
    ty = LSTM(64,return_sequences=True)(y)
    y = GlobalMaxPool1D()(y)
    t = Dropout(0.4)(y)
    y = Model(inputs=inputB, outputs=y)

    # combine the output of the two branches
    combined = keras.layers.concatenate([x.output, y.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs

    #z = Dense(128, activation="relu")(combined)
    z = Dense(5, activation="softmax")(combined)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.summary()
    return model

    ############GPU###################
    # model = multi_gpu_model(model, 2)
    optimizer = tf.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='CategoricalCrossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(model, X_train_tweet,X_train_topic, Y_train, X_val_tweet,X_val_topic, Y_val, filepath):
    """uncomment the comment below to perform the earlystoping strategy"""
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, mode='min', verbose=1,
                                              restore_best_weights=True)
    callbacks_list = [checkpoint, earlyStop]
    model.fit([X_train_tweet,X_train_topic], Y_train, validation_data=([X_val_tweet,X_val_topic], Y_val), epochs=100, batch_size=128, shuffle=True,
              callbacks=callbacks_list)


def plot_acc_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def split_data(x):
    tweet = np.zeros((len(x), len(x[0][0])))
    topic = np.zeros((len(x), len(x[0][1])))
    for i in range(len(x)):
        tweet[i] = x[i][0]
        topic[i] = x[i][1]
    return tweet,topic

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    X_merge,Y = data_load(data_root,label_root)

    embedding_model = load_w2v_model(w2v_root)
    embedding_layer = pretrained_layer(embedding_model)




    ################train#############################
    X_train, X_test, Y_train, Y_test = train_test_split(X_merge, Y, test_size=0.3, random_state=22, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=22, stratify=Y_test)

    X_train_tweet, X_train_topic = split_data(X_train)
    X_val_tweet, X_val_topic = split_data(X_val)
    X_test_tweet, X_test_topic = split_data(X_test)

    model = create_model(X_train_tweet, X_train_topic,embedding_layer)

    my_seed = 22
    np.random.seed(my_seed)

    #model.load_weights('./model/weights.best_biLSTM_ekphrasis.hdf5')
    optimizer = tf.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='CategoricalCrossentropy', optimizer=optimizer, metrics=['accuracy'])
    train_model(model, X_train_tweet,X_train_topic, Y_train, X_val_tweet,X_val_topic, Y_val, file_path)

    #model.fit([X_train_tweet,X_train_topic], Y_train, validation_data=([X_val_tweet,X_val_topic], Y_val), epochs = 200, batch_size = 128, shuffle=True,callbacks=callbacks_list)
    #plot_acc_loss(model.history)
    model.evaluate([X_test_tweet,X_test_topic], Y_test)




























