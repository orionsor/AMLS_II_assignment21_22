import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional,GlobalMaxPool1D,SpatialDropout1D,Conv1D,MaxPooling1D
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
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import os
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


w2v_root = "../Datasets/w2v_model/glove_twitter_200d.model"


def data_import():
    data1 = pd.read_table('../Datasets/A/english/twitter-2016train-A.txt' , usecols=[0,1,2], encoding='utf-8', names=['id','sentiment', 'tweet'])
    data2 = pd.read_table('../Datasets/A/english/twitter-2016test-A.txt' , usecols=[0,1,2], encoding='utf-8', names=['id','sentiment', 'tweet'])
    data = pd.concat([data1,data2],axis=0,ignore_index=True)
    return data

def plot_data_distribution(df):
    groups = df.groupby('label').count()   # beautiful graph
    plt.figure(figsize=(14, 12))
    groups['tweet'].plot(kind='bar')
    x = range(0,3,1)
    y = range(0,15000,2000)
    plt.xticks(x, ('negative', 'neutral',"positive"),rotation=0,weight='semibold')
    plt.yticks(y,weight='semibold')
    plt.tick_params(labelsize=12)
    plt.xlabel('Sentiment',fontdict={'weight':'semibold','size':16})
    plt.ylabel('Number of Tweets',fontdict={'weight':'semibold','size': 16})
    plt.title('Distribution of Sentiments before Augmentation',fontdict={'weight':'semibold','size': 20})
    plt.savefig('./plot/distribution_noaug_en.jpg')
    plt.show()

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
    for hashtag in map(lambda x: re.compile(re.escape(x)), [",", "\"", "=", "&", ";", "%", "$",
                                                            "@", "%", "^", "*", "(", ")", "{", "}",
                                                            "[", "]", "|", "/", "\\", "-",
                                                             ".", "'",
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
    #doc = remove_punctuations(doc)
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
    stop.append('may')
    stop.append('get')
    stop.append('amp')

    return stop



def data_tokenization(x):
    for i in range(len(x)):
        x[i] = nltk.word_tokenize(x[i])

    return x





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
        elif token in string.punctuation:
            cleaned_tokens.append(token)

    return cleaned_tokens

def data_to_le(x,stoplist):
  x = data_tokenization(x)
  temp = []
  for tokens in x:
    temp.append(lemmatize_sentence(tokens,stoplist))
  return temp

def load_w2v_model(root):
    w2v_model = KeyedVectors.load(root)
    return w2v_model

def cleared(word):
    res = ""
    prev = None
    for char in word:
        if char == prev: continue
        prev = char
        res += char
    return res

def create_data(data_pro,word_to_index,max_len):
    unks = []
    UNKS = []
    list_len = [len(i) for i in data_pro]
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




def create_Y(y):
  temp = np.zeros((len(y), ))
  for i  in range(len(y)):
    temp[i] = y[i]
  Y = to_categorical(temp,3)
  return Y



def eda_SR(originalSentence, n,stop):

    stops = stop
    splitSentence = list(originalSentence.split(" "))
    splitSentenceCopy = splitSentence.copy()
    # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
    ls_nonStopWordIndexes = []
    for i in range(len(splitSentence)):
        if splitSentence[i].lower() not in stops:
              ls_nonStopWordIndexes.append(i)
    if (n > len(ls_nonStopWordIndexes)):
        raise Exception("The number of replacements exceeds the number of non stop word words")
    for i in range(n):
        indexChosen = random.choice(ls_nonStopWordIndexes)
        ls_nonStopWordIndexes.remove(indexChosen)
        synonyms = []
        originalWord = splitSentenceCopy[indexChosen]
        for synset in nltk.corpus.wordnet.synsets(originalWord):
              for lemma in synset.lemmas():
                if lemma.name() != originalWord:
                      synonyms.append(lemma.name())
        if (synonyms == []):
              continue
        splitSentence[indexChosen] = random.choice(synonyms).replace('_', ' ')
    return " ".join(splitSentence)


def training_data_augmentation(x_train,y_train,n,stop):
    data_neg = []
    for i in range(len(x_train)):
        if y_train[i] == -1:
            data_neg.append(x_train[i])

    aug_neg = []
    for i in data_neg:
        aug_neg.append(eda_SR(i, n,stop))
    x_train.extend(aug_neg)
    for j in range(len(aug_neg)):
        y_train.append(-1)
    return x_train,y_train


def calculate_maxlen(x_train,x_val,x_test):
    list_len = [len(i) for i in x_train]
    max_len1 = max(list_len)
    list_len = [len(i) for i in x_val]
    max_len2 = max(list_len)
    list_len = [len(i) for i in x_test]
    max_len3 = max(list_len)
    max_len = max([max_len1, max_len2, max_len3])
    return max_len



if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    data = data_import()
    data = add_label(data)
    stemmer = ISRIStemmer()
    data["tweet"] = data['tweet'].apply(lambda x: processDocument(x, stemmer))

    data_list = data["tweet"] .tolist()
    label_list = data['label'].tolist()
    stoplist = stoplist_process()

    X_train, X_test, Y_train, Y_test = train_test_split(data_list, label_list, test_size=0.4, random_state=22,
                                                        stratify=label_list)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=22, stratify=Y_test)


    X_train, Y_train = training_data_augmentation(X_train,Y_train,3,stoplist)

    Y_train = create_Y(Y_train)
    Y_val = create_Y(Y_val)
    Y_test = create_Y(Y_test)

    X_train = data_to_le(X_train,stoplist)
    X_val = data_to_le(X_val,stoplist)
    X_test = data_to_le(X_test,stoplist)

    max_len = calculate_maxlen(X_train,X_val,X_test)
    embedding_model = load_w2v_model(w2v_root)
    X_train = create_data(X_train, embedding_model.key_to_index, max_len)
    X_val = create_data(X_val, embedding_model.key_to_index, max_len)
    X_test = create_data(X_test, embedding_model.key_to_index, max_len)
    pickle.dump(X_train, open("../Datasets/A/english/data_train_aug.p" , "wb"))
    pickle.dump(X_val, open("../Datasets/A/english/data_val_aug.p", "wb"))
    pickle.dump(X_test, open("../Datasets/A/english/data_test_aug.p", "wb"))
    pickle.dump(Y_train, open("../Datasets/A/english/label_train_aug.p", "wb"))
    pickle.dump(Y_val, open("../Datasets/A/english/label_val_aug.p", "wb"))
    pickle.dump(Y_test, open("../Datasets/A/english/label_test_aug.p", "wb"))













