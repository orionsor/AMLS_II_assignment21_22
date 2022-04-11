
import tensorflow as tf
import keras
from keras.models import Sequential
#from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional,GlobalMaxPool1D,SpatialDropout1D,Conv1D,MaxPooling1D
from keras.layers.embeddings import Embedding

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

from sklearn.model_selection import train_test_split


from joblib import dump, load
from nltk.stem.isri import ISRIStemmer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from time import time
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from tensorflow.keras.utils import to_categorical
import os
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')


#w2v_root = "./Datasets/w2v_model/glove_twitter_200d.model"
w2v_root = "./Datasets/w2v_model/glove_100d.model"

def data_import():
    data = pd.read_table('./Datasets/english/twitter-2016train-A.txt' , usecols=[0,1,2], encoding='utf-8', names=['id','sentiment', 'tweet'])
    return data

def add_label(df):
    CATEGORY_INDEX = {
    "negative": -1,
    "neutral": 0,
    "positive": 1
    }

    """transfer label into numeric data """
    raw_label = df['sentiment'].values.tolist()
    rawlabel = []
    for i in range(len(raw_label)):
        rawlabel.append(CATEGORY_INDEX[raw_label[i]])

    df['label'] = rawlabel

    return df

"""preprocess functions"""
def clean_base(tweets, clean_object):
    # tweets.loc[:, "tweet"].replace(clean_object, "", inplace=True)
    tweets = re.sub(clean_object, ' ', tweets)
    return tweets

def remove_urls(tweets):
    return clean_base(tweets, re.compile(r"http.?://[^\s]+[\s]?"))

def remove_usernames(tweets):
    return clean_base(tweets, re.compile(r"@[^\s]+[\s]?"))

def remove_hashtags(tweets):  # it unrolls the hashtags to normal words
    for hashtag in map(lambda x: re.compile(re.escape(x)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                            "@", "%", "^", "*", "(", ")", "{", "}",
                                                            "[", "]", "|", "/", "\\", ">", "<", "-",
                                                            "!", "?", ".", "'",
                                                            "--", "---", "#"]):
        tweets = re.sub(hashtag, ' ', tweets)
    return tweets

def remove_numbers(tweets):
    return clean_base(tweets, re.compile(r"\s?[0-9]+\.?[0-9]*"))

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def remove_punctuations(text):
    english_punctuations = string.punctuation
    punctuations_list = english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def processDocument(doc, stemmer):
    # Replace @username with empty string
    doc = remove_usernames(doc)
    # Replace url with empty string
    doc = remove_urls(doc)

    doc = re.sub(r'\n', ' ', doc)
    doc = re.sub(r'\d', '', doc)
    # Convert www.* or https?://* to " "
    doc = re.sub('(www\.[^\s])', ' ', doc)
    # Replace #word with word
    doc = re.sub(r'#([^\s]+)', r'\1', doc)

    # remove punctuations
    doc = remove_punctuations(doc)
    # normalize the tweet
    # doc= normalize_arabic(doc)

    # Replace numbers with empty string
    doc = remove_numbers(doc)
    # Replace @username with empty string
    doc = remove_hashtags(doc)

    # stemming
    doc = stemmer.stem(doc)
    return doc

def stoplist_process():
    stopwords = nltk.corpus.stopwords.words("english")
    whitelist = ["n't", "dn", "en", "tn", "not", "sn"]
    stop = []
    co = []
    for idx, stop_word in enumerate(stopwords):
        count = 0
        for whiteword in whitelist:
            if whiteword in stop_word:
                count += 1
        co.append(count)
        if not count != 0:
            stop.append(stop_word)

    return stop



def data_tokenization(df):
    Data = df['tweet'].tolist()
    for i in range(len(Data)):
        Data[i] = nltk.word_tokenize(Data[i])

    return Data



def lemmatize_sentence(tweet_tokens,STOP_WORDS):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        # Eliminating the token if it is a link

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token.lower(), pos)

        # Eliminating the token if its length is less than 3, if it is a punctuation or if it is a stopword
        if token not in string.punctuation and len(token) > 2 and token not in STOP_WORDS:
            cleaned_tokens.append(token)

    return cleaned_tokens

def load_w2v_model(root):
    w2v_model = KeyedVectors.load(root)
    return w2v_model


def pretrained_layer(embedding_model):
    word_dict = {}
    for word in list(embedding_model.index_to_key):
        word_dict[word] = embedding_model[word]


    Embedding_dim = embedding_model.vector_size

    word2idx = {'PAD': 0}
    # 所有词对应的嵌入向量 [(word, vector)]
    vocab_list = [word for word in enumerate(embedding_model.key_to_index.keys())]
    embeddings_matrix = np.zeros((len(vocab_list) + 1,embedding_model.vector_size))
    # word2idx 字典
    for i in range(len(vocab_list)):
        word = vocab_list[i][1]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = word_dict[word]

    # 初始化keras中的Embedding层权重
    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                      output_dim=Embedding_dim,
                      weights=[embeddings_matrix], # 预训练参数
                      trainable=False)
    return embedding_layer


def cleared(word):
    res = ""
    prev = None
    for char in word:
        if char == prev: continue
        prev = char
        res += char
    return res

def create_data(data_pro,word_to_index):
    unks = []
    UNKS = []
    list_len = [len(i) for i in data_pro]
    max_len = max(list_len)
    print('max_len:', max_len)

    X = np.zeros((len(data_pro), max_len))


    for i, tk_lb in enumerate(data_pro):
        tokens= tk_lb
        sentence_indices = []
        for j, w in enumerate(tokens):
            try:
                index = word_to_index[w]
            except:
                UNKS.append(w)
                w = cleared(w)
                try:
                    index = word_to_index[w]
                except:
                    index = word_to_index['unk']
                    unks.append(w)
            X[i, j] = index
    return X

def create_label(df,type = 'categorical'):

    if type == 'categorical':
        Y = np.zeros((len(df),))

        for i in range(len(df)):
            Y[i] = df['label'][i]
        Y = to_categorical(Y, 3)
    elif type == 'sparse':
        sparse_dic = {'-1': 0, '0': 1, '1': 2}
        Y = np.zeros((len(df),))
        for i in range(len(df)):
            Y[i] = sparse_dic['%d' % df['label'][i]]

    return Y

def create_model(embedding_layer):
    model = Sequential()

    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.4), merge_mode='concat'))
    model.add(Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.4), merge_mode='concat'))
    # model.add(LSTM(64,return_sequences=True))
    model.add(Conv1D(64, 4, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense((3), activation="softmax"))
    model.summary()

    ############GPU###################
    #model = multi_gpu_model(model, 2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='CategoricalCrossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(model,X_train, Y_train):
    """uncomment the comment below to perform the earlystoping strategy"""
    filepath = './Datasets/english/model/weights.best-glove200-lstm-cnn.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min', verbose=1,
                                              restore_best_weights=True)
    callbacks_list = [checkpoint, earlyStop]
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=128, shuffle=True,
              callbacks=callbacks_list)


def plot_acc_loss(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
    data = data_import()
    data = add_label(data)
    stemmer = ISRIStemmer()
    data["tweet"] = data['tweet'].apply(lambda x: processDocument(x, stemmer))
    stoplist = stoplist_process()
    data_tokenized = data_tokenization(data)
    data_processed = []
    # Removing noise from all the data
    for tokens in data_tokenized:
        data_processed.append(lemmatize_sentence(tokens,stoplist))

    embedding_model = load_w2v_model(w2v_root)
    embedding_layer = pretrained_layer(embedding_model)

    X = create_data(data_processed, embedding_model.key_to_index)
    Y = create_label(data,type = 'categorical')

    my_seed = 22
    np.random.seed(my_seed)
    model = create_model(embedding_layer)


    ################train#############################

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)
    train_model(model,X_train,Y_train)
    plot_acc_loss(model.history)

































