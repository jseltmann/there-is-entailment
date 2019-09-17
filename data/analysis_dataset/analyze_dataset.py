import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
nltk.download('stopwords')
import torch
import torch.nn.utils.rnn as rnn
import nltk
import csv
import pickle
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer

MAX_LEN = 25
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

INPUT_SIZE = 1
HIDDEN_SIZE = 25#100
with open("../../data/bert_classify_thereis_5caps_seed0/word_inds.pkl", "rb") as word_ind_file:
    word2num, _ = pickle.load(word_ind_file)
    PAD_INDEX = word2num["<PAD>"]

def load_data(train_filename, word_ind_filename, batch_size=64):
    """
    Load training data and translate the words to indices.

    Parameters
    ----------
    train_filename : str
        Filename of a csv file containg the training data.
    word_ind_filename : str
        Filename of a file containing dicts that translate
        from words to indices and from indices to words, respectively.
    batch_size : int
        batch size
    """

    tokenizer = RegexpTokenizer(r'\w+')
    inputs = []
    caps = []
    objs = []
    labels = []

    cap_freq_dict = defaultdict(int)
    obj_freq_dict = defaultdict(int)
    tog_freq_dict = defaultdict(int)
    syn_freq_dict = defaultdict(int)
    ola_freq_dict = defaultdict(int)

    with open(word_ind_filename, "rb") as word_ind_file:
        word2ind, ind2word = pickle.load(word_ind_file)

    objs = set()
    with open(train_filename) as train_file:
        train_reader = csv.reader(train_file, delimiter='\t', quotechar=None, escapechar="\\")
        for i, line in enumerate(train_reader):
            if i == 0:
                continue
            if len(line) < 4:
                continue
            obj = line[2].lower()
            objs.add(obj)
            
    with open(train_filename) as train_file:
        train_reader = csv.reader(train_file, delimiter='\t', quotechar=None, escapechar="\\")
        for i, line in enumerate(train_reader):
            if i == 0:
                continue
                
            if len(line) < 4:
                continue
                
            caption = line[1].lower()
            obj = line[2].lower()
            label = line[3]
            
            #print(caption, obj, label)
            
            cap_words = tokenizer.tokenize(caption)
            cap_words = cap_words[:MAX_LEN]

            obj_words = nltk.word_tokenize(obj)
            obj_words = obj_words[:MAX_LEN] # unlikely

            ignore_words = set(stopwords.words('english'))

            if label == "True":
                for word in cap_words:
                    if word not in ignore_words:
                        cap_freq_dict[word] += 1
                        for obj in obj_words:
                            tog_freq_dict[word+'_'+obj] += 1
                        for syn in wn.synsets(word):
                            syn_freq_dict[syn] += 1

            for obj in obj_words:
                ola_freq_dict[obj+str(label)] += 1
                
            if label == "True":
                for obj in obj_words:
                    obj_freq_dict[obj] += 1
            
            if label == "True":
                labels.append(torch.tensor([0.0,1.0]))
            elif label == "False":
                labels.append(torch.tensor([1.0,0.0]))
            else:
                raise Exception("label neither true nor false")

            if i % 50000 == 0:
                print("read", i, "training examples") 


    cap_freq = [(v, k) for k, v in cap_freq_dict.items()]
    cap_freq.sort(reverse=True)
    

    obj_freq = [(v, k) for k, v in obj_freq_dict.items()]
    obj_freq.sort(reverse=True)
    

    tog_freq = [(v, k) for k, v in tog_freq_dict.items()]
    tog_freq.sort(reverse=True)
    

    syn_freq = [(v, k) for k, v in syn_freq_dict.items()]
    syn_freq.sort(reverse=True)
    
    
    ola_pos_dict = defaultdict(int)
    ola_neg_dict = defaultdict(int)
    for obj in objs:
        ola_pos_dict[obj] = ola_freq_dict[obj+"True"]
        ola_neg_dict[obj] = ola_freq_dict[obj+"False"]
        
    ola_pos = [(v, k) for k, v in ola_pos_dict.items()]
    ola_pos.sort(reverse=True)
    
    ola_neg = [(v, k) for k, v in ola_neg_dict.items()]
    ola_neg.sort(reverse=True)
    
    return cap_freq, obj_freq, tog_freq, syn_freq, ola_pos, ola_neg
