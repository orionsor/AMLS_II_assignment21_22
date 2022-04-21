import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

'''this file cannot be run because necessary file is too large to be uploaded on Github
        only put here to show how the Glove model is generated in implementation'''

def create_w2v_model():
    glove_file = "./Datasets/glove/glove.6B.100d.txt"
    w2v_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

    return w2v_model

#w2v_model =create_w2v_model()
#w2v_model.save('./Datasets/w2v_model/glove_100d.model')