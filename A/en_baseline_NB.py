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

from time import time

import os
import pickle
from nltk.stem import SnowballStemmer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn import tree
from sklearn.svm import SVC



w2v_root = "../Datasets/w2v_model/glove_twitter_200d.model"

data_root = '../Datasets/A/english/data_ekphrasis.p'
label_root = '../Datasets/A/english/label_ekphrasis.p'

def data_load(data_root,label_root):

    data = pickle.load(open(data_root, "rb"))
    #label = pickle.load(open(label_root, "rb"))
    return data


def data_import():
    data1 = pd.read_table('../Datasets/A/english/twitter-2016train-A.txt' , usecols=[0,1,2], encoding='utf-8', names=['id','sentiment', 'tweet'])
    data2 = pd.read_table('../Datasets/A/english/twitter-2016test-A.txt' , usecols=[0,1,2], encoding='utf-8', names=['id','sentiment', 'tweet'])
    data = pd.concat([data1,data2],axis=0,ignore_index=True)
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


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target, y_pred, average='micro')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, X_train.shape[0]))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))



def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    fig = plt.figure(figsize = (15,13))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontdict={'weight':'semibold','size': 20})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontdict={'weight':'semibold','size':16})
    plt.xlabel('Predicted label', fontdict={'weight':'semibold','size':16})
    plt.savefig('./plot/cm_english.jpg')
    plt.show()



def calculate_results(y_true, y_pred):
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


def create_Y(y):
  temp = np.zeros((len(y), ))
  for i  in range(len(y)):
    temp[i] = y[i]
  #Y = to_categorical(temp,5)
  return temp



if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    X= data_load(data_root,label_root)
    data = data_import()
    data =add_label(data)
    label_list = data['label'].tolist()
    Y = create_Y(label_list)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=22, stratify=Y)


    clf_nbays = MultinomialNB()
    #clf_DT = tree.DecisionTreeClassifier(max_depth=500)
    #clf_svm = SVC()
    #train_predict(clf_nbays, X_train, Y_train, X_test, Y_test)
    clf_nbays.fit(X_train, Y_train)
    #Y_pred = clf_nbays.predict(X_test)
    Y_pred = clf_nbays.predict(X_test)
    Y_pred1 = clf_nbays.predict(X_train)
    result = calculate_results(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(cm, ['positive', 'neutral', 'negative'])
    print("F1 score for training set: {:.4f}.".format(f1_score(Y_train, Y_pred1,average='micro')))
    print("F1 score for test set: {:.4f}.".format(f1_score(Y_test, Y_pred,average='micro')))















