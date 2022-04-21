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
from nltk.stem.isri import ISRIStemmer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from time import time
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
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
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from wordcloud import WordCloud, STOPWORDS


w2v_root = "../Datasets/w2v_model/glove_twitter_200d.model"
root_1 = "../Datasets/C/english/twitter-2016train-CE.txt"
root_2 = "../Datasets/C/english/twitter-2016test-CE.txt"
def data_import(root1,root2):
    data1 = pd.read_table(root1, usecols=[0,1,2,3], encoding='utf-8', names=['id','topic','sentiment', 'tweet'])
    data2 = pd.read_table(root2 , usecols=[0,1,2,3], encoding='utf-8', names=['id','topic','sentiment', 'tweet'])
    data = pd.concat([data1,data2],axis=0,ignore_index=True)
    return data

def plot_data_distribution(df):
    groups = df.groupby('sentiment').count()  # beautiful graph
    plt.figure(figsize=(14, 12))
    groups['tweet'].plot(kind='bar')
    y = range(0,15000,2000)
     # beautiful graph
    groups['tweet'].plot(kind='bar')
    x = range(0, 5, 1)
    plt.xticks(x, ('very negative', 'negative', 'neutral', "positive", "very positive"),rotation=0,weight='semibold')
    plt.yticks(y,weight='semibold')
    plt.tick_params(labelsize=12)
    plt.xlabel('Sentiment',fontdict={'weight':'semibold','size':16})
    plt.ylabel('Number of Tweets',fontdict={'weight':'semibold','size': 16})
    plt.title('Distribution of Five-scale Sentiments',fontdict={'weight':'semibold','size': 20})
    plt.savefig('./plot/distribution__en.jpg')
    plt.show()



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

def processDocument(doc, stemmer,datatype ="tweet"):
    """summary of data cleansing"""
    """Replace @username with empty string"""
    if datatype == "tweet":
        doc = remove_usernames(doc)
    """Replace url with empty string"""
    doc = remove_urls(doc)

    doc = re.sub(r'\n', ' ', doc)
    doc = re.sub(r'\d', '', doc)
    """Convert www.* or https?://* to """
    doc = re.sub('(www\.[^\s])', ' ', doc)
    """Replace #word with word"""
    doc = re.sub(r'#([^\s]+)', r'\1', doc)

    """Replace numbers with empty string"""
    doc = remove_numbers(doc)
    """Replace selected hashtags with empty string"""
    doc = remove_hashtags(doc)

    # stemming
    doc = stemmer.stem(doc)
    return doc



def stoplist_process():
    """process of original stopwords"""
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

def create_tokenizer():
    """create tokenizer with ekphrasis"""
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "elongated",
                  'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )
    return text_processor

def data_tokenization(x,text_processor):
    for i in range(len(x)):
        x[i] = text_processor.pre_process_doc(x[i])

    return x




def lemmatize_sentence(tweet_tokens,STOP_WORDS):
    """recognize part of speech and restore word into original form"""
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):


        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token.lower(), pos)

        """Eliminating the token if its length is less than 2 or if it is a stopword"""
        if token not in string.punctuation and len(token) > 2 and token not in STOP_WORDS:
            cleaned_tokens.append(token)
        elif token in string.punctuation:
            cleaned_tokens.append(token)

    return cleaned_tokens

def data_to_le(x,stoplist,processer):
    """summary of tokenization and lemmatization"""
    x = data_tokenization(x,processer)
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

def create_data(data_pro,word_to_index):
    """convert text data into vectors"""
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


def create_Y(y):
  temp = np.zeros((len(y), ))
  for i  in range(len(y)):
    temp[i] = y[i]
  Y = to_categorical(temp,5)
  return Y




def create_sentiment_list(df,data_pro,sentiment = 'neutral'):
    senti = []
    CATEGORY_INDEX = {
        'very negative': -2,
        'negative': -1,
        'neutral': 0,
        "positive": 1,
        "very positive": 2
    }

    """Separating out positive and negative words (i.e., words appearing in negative and positive tweets),
                    in order to visualize each set of words seperately"""
    for i in range(len(df)):
        if df['sentiment'][i] == CATEGORY_INDEX[sentiment]:
            senti.extend(data_pro[i])
    return senti




def wordcloud_draw(data,stop,color='black', title='positive'):
    """draw word cloud based on occurrence frequency"""
    wordcloud = WordCloud(stopwords=stop,
                          background_color=color,
                          width=1500,
                          height=1000
                          ).generate(' '.join(data))
    plt.figure(1, figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('./plot/cloud_%s.jpg' % title)
    plt.show()








if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    data = data_import(root_1,root_2)
    #plot_data_distribution(data)
    stemmer = ISRIStemmer()
    data["tweet"] = data['tweet'].apply(lambda x: processDocument(x, stemmer,datatype="tweet"))
    data["topic"] = data['topic'].apply(lambda x: processDocument(x, stemmer,datatype="label"))
    stoplist = stoplist_process()
    data_list = data["tweet"] .tolist()
    topic_list = data['topic'].tolist()
    label_list = data['sentiment'].tolist()
    text_processor = create_tokenizer()
    data_processed = data_to_le(data_list,stoplist,text_processor)
    topic_processed = data_to_le(topic_list, stoplist,text_processor)

    positive_words = create_sentiment_list(data,data_processed,sentiment ='positive')
    very_positive_words = create_sentiment_list(data, data_processed, sentiment='very positive')
    neutral_words = create_sentiment_list(data, data_processed, sentiment='neutral')
    very_negative_words = create_sentiment_list(data, data_processed, sentiment='very negative')
    negative_words = create_sentiment_list(data, data_processed, sentiment='negative')
    wordcloud_draw(very_positive_words,stoplist, 'white', title='very positive')
    wordcloud_draw(positive_words, stoplist, 'white', title='positive')
    wordcloud_draw(neutral_words,stoplist, 'ghostwhite', title='neutral')
    wordcloud_draw(negative_words,stoplist, title='nagative')
    wordcloud_draw(very_negative_words,stoplist, title='very nagative')

    embedding_model = load_w2v_model(w2v_root)
    X = create_data(data_processed, embedding_model.key_to_index)
    X_topic = create_data(topic_processed, embedding_model.key_to_index)
    X_merge = []
    for i in range(len(X)):
        X_merge.append((X[i], X_topic[i]))
    Y = create_Y(label_list)
    pickle.dump(X_merge, open("../Datasets/C/english/data_noaug.p" , "wb"))
    pickle.dump(Y, open("../Datasets/C/english/label_noaug.p", "wb"))














