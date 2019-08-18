import pandas as pd
import numpy as np
import configparser
import os
import random
from textwrap import fill
import sys
from copy import deepcopy
import pickle
import random
import csv
import codecs
import nltk
import torch

random.seed(0)
MAX_LEN = 25

#config_path = os.environ.get('VISCONF')
#if not config_path:
#    # try default location, if not in environment
#    default_path_to_config = '../../../SwP/clp-vision/Config/gpu_potsdam.cfg'
#    if os.path.isfile(default_path_to_config):
#        config_path = default_path_to_config
#
#assert config_path is not None, 'You need to specify the path to the config file via environment variable VISCONF.'        
#
#config = configparser.ConfigParser()
#with open(config_path, 'r', encoding='utf-8') as f:
#    config.read_file(f)
#
#corpora_base = config.get('DEFAULT', 'corpora_base')
#preproc_path = config.get('DSGV-PATHS', 'preproc_path')
#dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')
#
#sys.path.append(dsgv_home + '/Utils')
#from utils import icorpus_code, plot_labelled_bb, get_image_filename, query_by_id
#from utils import plot_img_cropped, plot_img_ax, invert_dict, get_a_by_b
#sys.path.append(dsgv_home + '/WACs/WAC_Utils')
#from wac_utils import create_word2den, is_relational
#sys.path.append(dsgv_home + '/Preproc')
#from sim_preproc import load_imsim, n_most_sim
#
#sys.path.append('../../../SwP/sempix/Common')
#from data_utils import load_dfs, plot_rel_by_relid, get_obj_bb, compute_distance_objs
#from data_utils import get_obj_key, compute_relpos_relargs_row, get_all_predicate
#from data_utils import compute_distance_relargs_row, get_rel_type, get_rel_instances
#from data_utils import compute_obj_sizes_row
#
#df_names = ['mscoco_bbdf', 'refcoco_refdf', 'refcocoplus_refdf', 'grex_refdf',
#            'vgregdf', 'vgimgdf', 'vgobjdf', 'vgreldf',
#            'vgpardf', 'cococapdf']
#
#df = load_dfs(preproc_path, df_names)
#
#df['vgpregdf'] = df['vgregdf'][df['vgregdf']['pphrase'].notnull() & 
#                               (df['vgregdf']['pphrase'] != '')]
#coco_sem_sim, coco_sem_map = load_imsim(os.path.join(preproc_path, 'mscoco_sim.npz'))
#visg_sem_sim, visg_sem_map = load_imsim(os.path.join(preproc_path, 'visgen_sim.npz'))
#coco_id2semsim = invert_dict(coco_sem_map)
#visg_id2semsim = invert_dict(visg_sem_map)
#
#coco_vis_sim, coco_vis_map = load_imsim(os.path.join(preproc_path, 'mscoco_vis_sim.npz'))
#visg_vis_sim, visg_vis_map = load_imsim(os.path.join(preproc_path, 'visgen_vis_sim.npz'))
#coco_id2vissim = invert_dict(coco_vis_map)
#visg_id2vissim = invert_dict(visg_vis_map)
#
## intersecting visual genome and coco captions. Slow-ish.
#caption_coco_iids = list(set(df['cococapdf']['image_id'].tolist()))
## regions for only those image for which we also have coco captions
#visgencocap_regdf = df['vgregdf'].merge(pd.DataFrame(caption_coco_iids, columns=['coco_id']))
## coco_image_ids for images with both caption and region
#vgcap_coco_iids = list(set(visgencocap_regdf['coco_id'].tolist()))
## visgen_image_ids for images with both caption and region
#vgcap_vg_iids = list(set(visgencocap_regdf['image_id'].tolist()))
#
## map coco_ids to visgen_ids, and back
#coco2vg = dict(visgencocap_regdf[['coco_id', 'image_id']].values)
#vg2coco = dict([(v,k) for k,v in coco2vg.items()])
#
#df['vgpardf']['coco_image_id'] = df['vgpardf']['image_id'].apply(lambda x: vg2coco.get(x, None))
#df['cocoparcapdf'] = df['cococapdf'].merge(df['vgpardf'],
#                                           left_on='image_id', right_on='coco_image_id')

def create_binary_dataset(data_filename, sim_rank=0, train_split=[0.8,0.1,0.1]):
    """
    Create a datafile for the caption-to-there-is 
    entailment as binary classification.
    
    Parameters:
    -----------
    data_filename : str
        Filename to which to save the data.
    sim_rank : int
        Number of images from the vis_sim and sem_sim relations to ignore.
        E.g., if sim_rank is five, don't use objects from the five most similar
        images as negative examples.
    train_split : [float]
        Fractions of data to use for train, dev, and test set.
    """
    data_points = []

    phyps = set()
    nhyps = set()
    vgiis = set()
    i = 0

    for _, row in visgencocap_regdf.iterrows():
        vgii = row['image_id']
        if vgii in vgiis:
            continue

        if i == 10:
            print("10")
        if i == 100:
            print("100")
        if i % 5000 == 0:
            print("processed", i, "rows ...")
        vgiis.add(vgii)
        i += 1

        cocoii = row['coco_id']

        # get objects in image
        obj_syns = set(df['vgobjdf'][df['vgobjdf']['image_id'] == vgii]['syn'].tolist())
        if None in obj_syns:
            #if an object has no syn_id ignore this caption
            continue

        curr_phyps = set(df['vgobjdf'][df['vgobjdf']['image_id'] == vgii]['name'].tolist())
        curr_phyps = [(vgii, phyp, True) for phyp in curr_phyps]
        phyps.update(curr_phyps)

        try:
            ignore_sem_ids = n_most_sim(visg_sem_sim, visg_sem_map,
                                        visg_id2semsim[vgii], n=sim_rank)
            ignore_vis_ids = n_most_sim(visg_vis_sim, visg_vis_map,
	                                visg_id2vissim[vgii], n=sim_rank)
            ignore_ids = ignore_vis_ids + ignore_sem_ids
            ignore_ids.append(vgii)
        except Exception as e:
            ignore_ids = []

        neg_num = len(curr_phyps)
        curr_nhyps = set()
        while len(curr_nhyps) < neg_num:
            neg_sample = df['vgobjdf'].sample()
            negii = neg_sample['image_id'].values[0]
            if negii in ignore_ids:
                continue
            neg_obj = neg_sample['syn'].values[0]
            if neg_obj in obj_syns:
                continue
            neg_name = neg_sample['name'].values[0]
            curr_nhyps.add((vgii, neg_name, False))
        nhyps.update(curr_nhyps)

    print("found hypotheses")

    phyps = list(phyps)
    nhyps = list(nhyps)
    np.random.shuffle(phyps)
    np.random.shuffle(nhyps)

    plen = len(phyps)
    train_len = int(train_split[0] * plen)
    dev_len = int(train_split[1] * plen)

    train_hyps = phyps[:train_len] + nhyps[:train_len]
    np.random.shuffle(train_hyps)
    dev_hyps = phyps[train_len:(train_len+dev_len)] + nhyps[train_len:(train_len+dev_len)]
    np.random.shuffle(dev_hyps)
    test_hyps = phyps[train_len+dev_len:] + nhyps[train_len+dev_len:]
    np.random.shuffle(test_hyps)

    data = dict()
    data['train_hyps'] = train_hyps
    data['dev_hyps'] = dev_hyps
    data['test_hyps'] = test_hyps

    with open(data_filename, "wb") as data_file:
        pickle.dump(data, data_file)

def preproc_bert_baseline(data_filename, bert_data_path, num_captions=5):
    """
    Write the data as tsv files, so that it can be used for BERT finetuning.

    Parameters
    ----------
    data_filename: str
        Path to the file created by create_binary_dataset().
    bert_data_path: str
        Directory into which to put the tsv files.
    num_captions: int
        Number of captions per image to include in data.
        (Since COCO contains five captions per image.)
    """

    if num_captions > 5:
        num_captions = 5
    if num_captions < 1:
        num_captions = 1

    with open(data_filename, "rb") as data_file:
        data = pickle.load(data_file)

    for split_name, examples in data.items():
        tsv_filename = os.path.join(bert_data_path, split_name[:-5] + ".tsv")
        
        with open(tsv_filename, "w") as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t', quotechar=None, escapechar="\\")
            #tsv_file.write("index\tcaption\tobject\tentailment\n")
            tsv_writer.writerow(["index", "caption", "object", "entailment"])

            counter = 0
            for index, (image_id, obj, entailment) in enumerate(examples):
                #extract caption
                if len(obj.split('\x00')) > 1:
                    obj = "".join(obj.split('\x00'))
                visgen_rows = visgencocap_regdf[visgencocap_regdf['image_id'] == image_id]
                coco_id = int(visgen_rows.sample()["coco_id"].values[0])
                this_df = df["cococapdf"]
                caps = this_df[this_df['image_id'] == coco_id]['caption'].values
                for cap in caps:
                    if len(cap.split("\x00")) > 1:
                        cap = "".join(cap.split('\x00'))
                caps_to_use = caps[:num_captions]


                for cap in caps_to_use:
                    tsv_writer.writerow([counter, cap, obj, entailment])
                    counter += 1
                if index % 10000 == 0:
                    print("processed", index, split_name[:-5], "examples")
        print("wrote", split_name[:-5], "file")


def create_generation_dataset(data_path, train_split=[0.8,0.1,0.1]):
    """
    Create a datafile for the caption-to-there-is 
    entailment as generation task.
    
    Parameters:
    -----------
    data_path : str
        Directory to which to save the data.
    train_split : [float]
        Fractions of data to use for train, dev, and test set.
    """
    vgiis = set()
    i = 0
    examples = []

    for _, row in visgencocap_regdf.iterrows():
        vgii = row['image_id']
        if vgii in vgiis:
            continue

        if i == 10:
            print("10")
        if i == 100:
            print("100")
        if i % 5000 == 0:
            print("processed", i, "rows ...")
        vgiis.add(vgii)
        i += 1

        cocoii = row['coco_id']

        objects = set(df['vgobjdf'][df['vgobjdf']['image_id'] == vgii]['name'].tolist())
        
        this_df = df["cococapdf"]
        caps = this_df[this_df['image_id'] == cocoii]['caption'].values
        

        curr_hyps = (caps, objects)
        examples.append(curr_hyps)
        


    print("found hypotheses")

    num_rows = len(examples)
    train_len = int(train_split[0] * num_rows)
    dev_len = int(train_split[1] * num_rows)

    train_hyps = examples[:train_len]
    np.random.shuffle(train_hyps)
    dev_hyps = examples[train_len:(train_len+dev_len)]
    np.random.shuffle(dev_hyps)
    test_hyps = examples[train_len+dev_len:]
    np.random.shuffle(test_hyps)

    #write train examples
    for name, split in zip(["train","dev","test"],[train_hyps, dev_hyps, test_hyps]):
        tsv_filename = os.path.join(data_path, name + ".tsv")
        with open(tsv_filename, "w") as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t', quotechar=None, escapechar="\\")
            tsv_writer.writerow(["index","caption","objects"])

            index = 0
            for caps, objs in split:
                for cap in caps:
                    tsv_writer.writerow([index, cap, objs])
                    index += 1
        print("wrote", name, "examples")


def create_word_ind_dict(data_path):
    """
    Make a numeric index for every(!) word in the data,
    to be used in the LSTM.

    Parameters
    ----------
    data_path : str
        Path containing csv files created by create_generation_dataset or preproc_bert_baseline.
        The resulting word indices will also be saved there.
    """
    vocab = set()
    vocab.add("<START>")
    vocab.add("<END>")
    vocab.add("<UNK>")
    vocab.add("<PAD>")
 
    for name in ["train", "dev", "test"]:
        tsv_filename = os.path.join(data_path, name + ".tsv")
        with open(tsv_filename, "r") as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t", quotechar=None, escapechar="\\")
            i = 0
            for line in tsv_reader:
                if i == 0:
                    i += 1
                    continue
                for word in nltk.word_tokenize(line[1]):
                    vocab.add(word.lower())
                for word in nltk.word_tokenize(line[2]):
                    vocab.add(word.lower())
                if i % 100000 == 0 or i == 1000:
                    print("processed", i, "examples")
                i += 1

    word2ind = dict()
    ind2word = dict()
    for ind, word in enumerate(vocab):
        word2ind[word] = ind
        ind2word[ind] = word

    ind_path = os.path.join(data_path, "word_inds.pkl")
    with open(ind_path, "wb") as ind_file:
        #pickle.dump(ind_file, (word2ind, ind2word))
        pickle.dump((word2ind, ind2word), ind_file)


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

    inputs = []
    caps = []
    objs = []
    labels = []

    with open(word_ind_filename, "rb") as word_ind_file:
        word2ind, ind2word = pickle.load(word_ind_file)

    with open(train_filename) as train_file:
        #train_reader = csv.reader(train_file, delimiter='\t', quotechar=None, escapechar="\\")
        train_reader = csv.reader(train_file, delimiter='\t', escapechar="\\")
        for i, line in enumerate(train_reader):
            if i == 122467:
                print(line)
            if i == 0:
                continue
            caption = line[1].lower()
            obj = line[2].lower()
            label = line[3]
            
            cap_words = nltk.word_tokenize(caption)
            cap_words = cap_words[:MAX_LEN]
            cap_inds = torch.tensor([float(word2ind[word]) for word in cap_words])
            caps.append(cap_inds)

            obj_words = nltk.word_tokenize(obj)
            obj_words = obj_words[:MAX_LEN] # unlikely
            obj_inds = torch.tensor([float(word2ind[word]) for word in obj_words])
            objs.append(obj_inds)

            if label == "True":
                labels.append(torch.tensor([0.0,1.0]))
            elif label == "False":
                labels.append(torch.tensor([1.0,0.0]))
            else:
                print("###" + label + "###")
                print(line)
                print(i)
                raise Exception("label neither true nor false")

            if i % 50000 == 0:
                print("read", i, "training examples") 

    caps_batched = []
    while caps != []:
        caps_batched.append(caps[:batch_size])
        caps = caps[batch_size:]

    objs_batched = []
    while objs != []:
        objs_batched.append(objs[:batch_size])
        objs = objs[batch_size:]

    labels_batched = []
    while labels != []:
        labels_batched.append(labels[:batch_size])
        labels = labels[batch_size:]

    return caps_batched, objs_batched, labels_batched

word_ind_filename = "/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/cap_not_seen/word_inds.pkl"
caps, objs, labels = load_data("/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/cap_not_seen/train.tsv", 
                               word_ind_filename, 
                               batch_size=64)

with open("/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/cap_not_seen/lstm_preprocessed_train.pkl", "wb") as processed_file:
    pickle.dump((caps,objs,labels), processed_file)


caps, objs, labels = load_data("/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/cap_not_seen/dev.tsv", 
                               word_ind_filename, 
                               batch_size=64)

with open("/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/cap_not_seen/lstm_preprocessed_dev.pkl", "wb") as processed_file:
    pickle.dump((caps,objs,labels), processed_file)


#create_word_ind_dict("../../../data/entailment_data_analysis/obj_in_caption/cap_not_seen/")
#preproc_bert_baseline("../../data/binary_class.pkl", "../../data/bert_classify_thereis_5caps", num_captions=5)
