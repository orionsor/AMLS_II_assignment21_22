
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
import string

from nltk.stem.isri import ISRIStemmer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from tensorflow.keras.utils import to_categorical
import os
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from wordcloud import WordCloud, STOPWORDS

w2v_root = "../Datasets/w2v_model/glove_twitter_200d.model"
"""this file is to generate the data through preprocess without data augmentation and the assistance of ekphrasis library"""

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
    """replace object to be cleaned with space"""
    tweets = re.sub(clean_object, ' ', tweets)
    return tweets

def remove_urls(tweets):
    return clean_base(tweets, re.compile(r"http.?://[^\s]+[\s]?"))

def remove_usernames(tweets):
    return clean_base(tweets, re.compile(r"@[^\s]+[\s]?"))

def remove_hashtags(tweets):
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
    """summary of data cleansing"""
    """Replace @username with empty string"""
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

    """stemming"""
    doc = stemmer.stem(doc)
    return doc

def stoplist_process():
    """process of original stopwords"""
    stopwords = nltk.corpus.stopwords.words("english")
    """reserve negative words which can alter sentiments"""
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
        x[i] = nltk.word_tokenize(x[i])

    return x


def lemmatize_sentence(tweet_tokens,STOP_WORDS):
    """recognize part of speech and restore word into original form"""
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

        """Eliminating the token if its length is less than 2 or if it is a stopword"""
        if token not in string.punctuation and len(token) > 2 and token not in STOP_WORDS:
            cleaned_tokens.append(token)
        elif token in string.punctuation:
            cleaned_tokens.append(token)

    return cleaned_tokens

def data_to_le(x,stoplist):
    """summary of tokenization and lemmatization"""
    x = data_tokenization(x)
    temp = []
    for tokens in x:
        temp.append(lemmatize_sentence(tokens,stoplist))
    return temp

def load_w2v_model(root):
    """load GloVe corpus and model"""
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




def create_sentiment_list(df,data_pro,sentiment = 'neutral'):
    senti = []

    '''Separating out positive and negative words (i.e., words appearing in negative and positive tweets),
                                  in order to visualize each set of words seperately'''
    for i in range(len(df)):
        if df['sentiment'][i] == sentiment:
            senti.extend(data_pro[i])
    return senti



def wordcloud_draw(data,stop,color='black', title='positive'):
    """draw word cloud"""
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
    data = data_import()
    data = add_label(data)
    #plot_data_distribution(data)
    stemmer = ISRIStemmer()
    data["tweet"] = data['tweet'].apply(lambda x: processDocument(x, stemmer))
    stoplist = stoplist_process()
    data_list = data["tweet"] .tolist()
    data_processed = data_to_le(data_list,stoplist)

    positive_words = create_sentiment_list(data,data_processed,sentiment ='positive')
    neutral_words = create_sentiment_list(data, data_processed, sentiment='neutral')
    negative_words = create_sentiment_list(data, data_processed, sentiment='negative')
    wordcloud_draw(positive_words,stoplist, 'white', title='positive')
    wordcloud_draw(neutral_words,stoplist, 'ghostwhite', title='neutral')
    wordcloud_draw(negative_words,stoplist, title='nagative')

    embedding_model = load_w2v_model(w2v_root)
    X = create_data(data_processed, embedding_model.key_to_index)
    Y = create_label(data,type = 'categorical')
    pickle.dump(X, open("../Datasets/A/english/data_noaug.p" , "wb"))
    pickle.dump(Y, open("../Datasets/A/english/label_noaug.p", "wb"))














