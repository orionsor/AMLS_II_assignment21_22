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
from nltk.stem.isri import ISRIStemmer
from collections import Counter
import itertools
import string
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,precision_score

from time import time

import os
import pickle


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer




"""this file is to build, train, validate and evaluate Naive Bayes model
      based on the data through preprocess without the assistance of ekphrasis library.
      this file serves as a baseline for comparison"""

w2v_root = "../Datasets/w2v_model/glove_twitter_200d.model"

data_root = '../Datasets/A/english/data_ekphrasis.p'
label_root = '../Datasets/A/english/label_ekphrasis.p'

def data_load(data_root):

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
def clean_base(tweets, clean_object):
    """replace object to be cleaned with space"""
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



def processDocument(doc):
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

    # Replace numbers with empty string
    doc = remove_numbers(doc)
    # Replace hashtags with empty string
    doc = remove_hashtags(doc)

    # stemming
    #doc = stemmer.stem(doc)
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
    day = ['tomorrow', 'yesterday', 'today', 'day', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
           'saturday']
    number = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    stop.append('may')
    stop.append('get')
    stop.append('amp')
    stop.extend(day)
    stop.extend(number)

    return stop

def data_tokenization(x):
    for i in range(len(x)):
        x['tweet'][i] = nltk.word_tokenize(x['tweet'][i])

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

        # Eliminating the token if its length is less than 2 if it is a punctuation or if it is a stopword
        if token not in string.punctuation and len(token) > 2 and token not in STOP_WORDS:
            cleaned_tokens.append(token)
        elif token in string.punctuation:
            cleaned_tokens.append(token)

    return cleaned_tokens

def data_to_le(x,stoplist):
  x["tweet"] = x["tweet"].apply(nltk.word_tokenize)
  temp = []
  #x['tweet'] = x['tweet'].apply(lambda k: [item for item in k if item not in stoplist])
  data_pro = []
  for i in range(len(x)):
      temp = []
      for tokens in x['tweet'][i] :
          if len(tokens) > 2 and tokens not in stoplist:
              temp.append(tokens)
      data_pro.append(str(temp))

  return data_pro

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

    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target, y_pred, average='micro')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    '''Indicate the classifier and the training set size'''
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, X_train.shape[0]))

    '''Train the classifier'''
    train_classifier(clf, X_train, y_train)

    '''Print the results of prediction for both training and testing'''
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

    X= data_load(data_root)
    data = data_import()
    data =add_label(data)
    label_list = data['label'].tolist()
    Y = create_Y(label_list)
    data["tweet"] = data['tweet'].apply(lambda x: processDocument(x))
    stoplist = stoplist_process()
    data_processed = data_to_le(data,stoplist)


    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), strip_accents='unicode', norm='l2')
    X = vectorizer.fit_transform(data_processed)
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
    plot_confusion_matrix(cm, ['negative', 'neutral', 'positive'])


    print("F1 score for training set: {:.4f}.".format(f1_score(Y_train, Y_pred1,average='macro')))
    print("F1 score for test set: {:.4f}.".format(f1_score(Y_test, Y_pred,average='macro')))















