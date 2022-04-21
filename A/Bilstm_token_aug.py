import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GlobalMaxPool1D, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import os
import random
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import os
import pickle

w2v_root = "../Datasets/w2v_model/glove_twitter_200d.model"
data_root = '../Datasets/A/english/'

"""this file is to build, train, validate and evaluate Bi-LSTM model
      based on the augmented data through preprocess with the assistance of ekphrasis library"""

def data_load(root,datasets = 'train',datatype='data'):
    """load data generated in advance if run this script directly"""
    filepath_data = os.path.join(root, '%s_%s_aug.p'%(datatype,datasets))
    data = pickle.load(open(filepath_data, "rb"))

    return data



def load_w2v_model(root):
    """load pretrained GloVe word to vector model"""
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
                                weights=[embeddings_matrix],  # 预训练参数
                                trainable=False)
    return embedding_layer




def create_model(embedding_layer):

    model = Sequential()

    model.add(embedding_layer)
    #model.add(GaussianNoise(0.2))
    model.add(Dropout(0.3))
    model.add((LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.4))
    # model.add(Dense(64, activation="relu"))
    # model.add(Dropout(0.4))
    # model.add(Dense((3), activation="softmax",kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense((3), activation="softmax"))

    model.summary()

    ############GPU###################
    # model = multi_gpu_model(model, 2)
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='CategoricalCrossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(model, X_train, Y_train, X_val, Y_val):
    """uncomment the comment below to perform the earlystoping strategy"""
    filepath = './model/weights.best-lstm-aug.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, mode='max', verbose=1,
                                              restore_best_weights=True)
    callbacks_list = [checkpoint, earlyStop]
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=256, shuffle=True,
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


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    X_train = data_load(data_root,datasets='train',datatype='data')
    X_test = data_load(data_root, datasets='test', datatype='data')
    X_val = data_load(data_root, datasets='val', datatype='data')
    Y_train = data_load(data_root, datasets='train', datatype='label')
    Y_test = data_load(data_root, datasets='test', datatype='label')
    Y_val = data_load(data_root, datasets='val', datatype='label')


    embedding_model = load_w2v_model(w2v_root)
    embedding_layer = pretrained_layer(embedding_model)


    my_seed = 22
    np.random.seed(my_seed)
    model = create_model(embedding_layer)
    #model.load_weights('./model/weights.best_token_biLSTM_aug.hdf5')

    ################train#############################

    train_model(model, X_train, Y_train, X_val, Y_val)
    plot_acc_loss(model.history)
    model.evaluate(X_test, Y_test)
































