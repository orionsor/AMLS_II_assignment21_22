from A import data_process_ekphrasis
from A import BiLSTM
from C import c_data_process
from C import c_biLSTM_ek
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.stem.isri import ISRIStemmer
import tensorflow as tf

A_root1 = "./Datasets/A/english/twitter-2016train-A.txt"
A_root2 = "./Datasets/A/english/twitter-2016test-A.txt"
w2v_root = "./Datasets/w2v_model/glove_twitter_200d.model"
C_root1 = "./Datasets/C/english/twitter-2016train-CE.txt"
C_root2 = "./Datasets/C/english/twitter-2016test-CE.txt"
filepath_A = './Datasets/A/english/model/weights.best-lstm.hdf5'
filepath_C = './Datasets/C/english/model/weights.best-lstm-c.hdf5'
# ======================================================================================================================
# Data preprocessing-A

data_A = data_process_ekphrasis.data_import(A_root1, A_root2)
data_A = data_process_ekphrasis.add_label(data_A)
data_A["tweet"] = data_A['tweet'].apply(lambda x: data_process_ekphrasis.processDocument(x))
stoplist = data_process_ekphrasis.stoplist_process()
data_list = data_A["tweet"].tolist()
text_processor = data_process_ekphrasis.create_tokenizer()
data_processed = data_process_ekphrasis.data_to_le(data_list, stoplist, text_processor)
embedding_model = data_process_ekphrasis.load_w2v_model(w2v_root)
X_A = data_process_ekphrasis.create_data(data_processed, embedding_model.key_to_index)
Y_A = data_process_ekphrasis.create_label(data_A, type='categorical')
X_train_A, X_test_A, Y_train_A, Y_test_A = train_test_split(X_A, Y_A, test_size=0.4, random_state=17, stratify=Y_A)
X_val_A, X_test_A, Y_val_A, Y_test_A = train_test_split(X_test_A, Y_test_A, test_size=0.5, random_state=17, stratify=Y_test_A)

# ======================================================================================================================
# Task A
embedding_model = BiLSTM.load_w2v_model(w2v_root)
embedding_layer = BiLSTM.pretrained_layer(embedding_model)


model_A = BiLSTM.create_model(embedding_layer)
#model_A.load_weights('./A/model/weights.best_biLSTM.hdf5')
BiLSTM.train_model(model_A, X_train_A, Y_train_A, X_val_A, Y_val_A,filepath_A)
################train#############################
#plot_acc_loss(model.history)
_,acc_A_train =model_A.evaluate(X_train_A, Y_train_A)
_,acc_A_test =model_A.evaluate(X_test_A, Y_test_A)


# ======================================================================================================================
# Data preprocessing-C
data_C = c_data_process.data_import(C_root1, C_root2)
stemmer = ISRIStemmer()
data_C["tweet"] = data_C['tweet'].apply(lambda x: c_data_process.processDocument(x, stemmer, datatype="tweet"))
data_C["topic"] = data_C['topic'].apply(lambda x: c_data_process.processDocument(x, stemmer, datatype="label"))
stoplist = c_data_process.stoplist_process()
data_list_C = data_C["tweet"].tolist()
topic_list = data_C['topic'].tolist()
label_list_C = data_C['sentiment'].tolist()
text_processor = c_data_process.create_tokenizer()
data_processed_C = c_data_process.data_to_le(data_list_C, stoplist, text_processor)
topic_processed = c_data_process.data_to_le(topic_list, stoplist, text_processor)
embedding_model = c_data_process.load_w2v_model(w2v_root)
X_C = c_data_process.create_data(data_processed_C, embedding_model.key_to_index)
X_topic = c_data_process.create_data(topic_processed, embedding_model.key_to_index)
X_merge = []
for i in range(len(X_C)):
    X_merge.append((X_C[i], X_topic[i]))
Y_C = c_data_process.create_Y(label_list_C)
# ======================================================================================================================
# Task C

embedding_model = c_biLSTM_ek.load_w2v_model(w2v_root)
embedding_layer = c_biLSTM_ek.pretrained_layer(embedding_model)

################train#############################
X_train_C, X_test_C, Y_train_C, Y_test_C = train_test_split(X_merge, Y_C, test_size=0.4, random_state=22, stratify=Y_C)
X_val_C, X_test_C, Y_val_C, Y_test_C = train_test_split(X_test_C, Y_test_C, test_size=0.5, random_state=22, stratify=Y_test_C)

X_train_tweet, X_train_topic = c_biLSTM_ek.split_data(X_train_C)
X_val_tweet, X_val_topic = c_biLSTM_ek.split_data(X_val_C)
X_test_tweet, X_test_topic = c_biLSTM_ek.split_data(X_test_C)

model_C = c_biLSTM_ek.create_model(X_train_tweet, X_train_topic, embedding_layer)
optimizer = tf.optimizers.Adam(learning_rate=0.0005)
model_C.compile(loss='CategoricalCrossentropy', optimizer=optimizer, metrics=['accuracy'])
c_biLSTM_ek.train_model(model_C, X_train_tweet,X_train_topic, Y_train_C, X_val_tweet,X_val_topic, Y_val_C, filepath_C)


_,acc_B_train = model_C.evaluate([X_train_tweet,X_train_topic], Y_train_C)
_,acc_B_test = model_C.evaluate([X_test_tweet,X_test_topic], Y_test_C)






# ======================================================================================================================
## Print out your results with following format:

print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
                                                        acc_B_train, acc_B_test))


