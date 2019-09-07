import torch
import pickle
import torch.nn.utils.rnn as rnn
import numpy as np
import csv
from scipy.spatial.distance import cosine
import nltk
import sklearn.metrics as metrics
import os, os.path


def smaller_set(dev_path, num_examples):
    """
    Create a smaller set, for faster evaluation.
    """

    true_count = 0
    false_count = 0

    with open(dev_path) as dev_file:
        reader = csv.reader(dev_file, delimiter="\t", quotechar=None, escapechar="\\")
        with open(dev_path[:-4] + "_" + str(num_examples) + ".tsv", "w") as short_file:
            writer = csv.writer(short_file, delimiter="\t", quotechar=None, escapechar="\\")
            for i, line in enumerate(reader):
                if i == 0:
                    writer.writerow(line)
                label = line[3]
                if label == "True" and true_count < num_examples / 2:
                    writer.writerow(line)
                    true_count += 1
                if label == "False" and false_count < num_examples / 2:
                    writer.writerow(line)
                    false_count += 1
                if false_count + true_count >= num_examples:
                    break


def load_vectors(vector_filename, num_vecs=10000):
    """
    Load glove vectors.

    Parameters
    ----------
    vector_filename : str
        File to read the vectors from.
    num_vecs : int
        Number of vectors to use.

    Return
    ------
    vectors : dict[str]
        Dict containing the vector for each word.
    """

    vectors = dict()

    with open(vector_filename) as vector_file:
        for i, line in enumerate(vector_file):
            entries = line.split()
            word = entries[0]
            nums = entries[1:]
            nums = [float(num) for num in nums]
            vector = np.array(nums)
            vectors[word] = vector

    return vectors


def load_objects(train_filename):
    """
    Load all possible objects from the train corpus.

    Parameters
    ----------
    train_filename : str
        Path of tsv file containing the train set.
    
    Return
    ------
    objects : set(str)
        Set of objects in the training set.
    """
    
    with open(train_filename) as train_file:
        train_reader = csv.reader(train_file, delimiter="\t", quotechar=None, escapechar="\\")
        
        objs = set()

        for i, line in enumerate(train_reader):
            if i == 0:
                continue
            
            obj = line[2]
            objs.add(obj)

    return objs


def evaluate_vector_addition(vectors, objects, dev_path, log_path, cutoff=40, vector_path=None, stopwords=[]):
    """
    Evaluate LSTM model by calculating the accuracy over a test set.

    Parameters
    ----------
    vectors : dict[str]
        Dict containing the vector for each word.
    objects : set(str)
        Objects in the train set.
    dev_path : str
        Path of file containing the set on which to evaluate the model.
    log_path : str
        Path to write the results to.
    vector_path : str
        Path to file containing vectors.
    stopwords : [str]
        List of words to ignore in the addition.
    """

    vector_shape = list(vectors.values())[0].shape

    obj_vectors = []
    for obj in objects:
        vec = np.zeros(vector_shape)
        for word in nltk.word_tokenize(obj):
            word = word.lower()
            if word in vectors:
                vec += vectors[word]
        obj_vectors.append(vec)

    labels = []
    preds = []
    labels_unk_obj = []
    preds_unk_obj = []

    with open(dev_path) as dev_file:
        dev_reader = csv.reader(dev_file, delimiter="\t", quotechar=None, escapechar="\\")
        for i, line in enumerate(dev_reader):
            if i == 0:
                continue
            #if i > 100 and i <= 67310:
            #    continue
            #if i > 67410:
            #    break
            try:
                _, cap, obj, label = line #line.split("\t")
            except Exception as e:
                print(line)
                7 / 0
            cap = cap.split()
            obj_list = obj.split()
            if label == "True":
                label = True
            else:
                label = False
            labels.append(label)
    
            cap_sum = np.zeros(vector_shape)
            for word in cap:
                word = word.lower()
                if word in vectors:
                    cap_sum += vectors[word]
    
            obj_sum = np.zeros(vector_shape)
            for word in obj_list:
                word = word.lower()
                if word in vectors and not word in stopwords:
                    obj_sum += vectors[word]
    
            obj_dist = cosine(cap_sum, obj_sum)
    
            #word_dists = []
            closer_count = 0
            #for j, word in enumerate(vectors):
            for vec in obj_vectors:
                curr_dist = cosine(cap_sum, vec)
                if curr_dist < obj_dist:
                    closer_count += 1
                if closer_count > cutoff:
                    break
                #word_dists.append(curr_dist)
                #if j % 50000 == 0 or j == 1 or j == 100:
                #    print(j)
            #closer = [dist for dist in word_dists if dist < obj_dist]

            #if len(closer) > cutoff:
            if closer_count > cutoff:
                pred = False
            else:
                pred = True

            preds.append(pred)
            
            if len(log_path) > 4:
                log_dir = log_path[:-4]
            else:
                log_dir = log_path + "_dir"

            line = " ".join(line) + "\n"

            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            with open(log_dir + "/true_pos.log", "w") as true_pos:
                tp_writer = csv.writer(true_pos, escapechar="\\", delimiter="\t")
                with open(log_dir + "/false_pos.log", "w") as false_pos:
                    fp_writer = csv.writer(false_pos, escapechar="\\", delimiter="\t")
                    with open(log_dir + "/true_neg.log", "w") as true_neg:
                        tn_writer = csv.writer(true_neg, escapechar="\\", delimiter="\t")
                        with open(log_dir + "/false_neg.log", "w") as false_neg:
                            fn_writer = csv.writer(false_neg, escapechar="\\", delimiter="\t")
                            for (pred, label) in zip(preds, labels):
                                if pred == True and label == True:
                                    tp_writer.writerow(line)
                                elif pred == True and label == False:
                                    fp_writer.writerow(line)
                                elif pred == False and label == False:
                                    tn_writer.writerow(line)
                                else:
                                    fn_writer.writerow(line)

            if obj not in objects:
                preds_unk_obj.append(pred)
                labels_unk_obj.append(label)
                with open(log_dir + "/true_pos_unk_obj.log", "w") as true_pos:
                    tp_writer = csv.writer(true_pos, escapechar="\\", delimiter="\t")
                    with open(log_dir + "/false_pos_unk_obj.log", "w") as false_pos:
                        fp_writer = csv.writer(false_pos, escapechar="\\", delimiter="\t")
                        with open(log_dir + "/true_neg_unk_obj.log", "w") as true_neg:
                            tn_writer = csv.writer(true_neg, escapechar="\\", delimiter="\t")
                            with open(log_dir + "/false_neg_unk_obj.log", "w") as false_neg:
                                fn_writer = csv.writer(false_neg, escapechar="\\", delimiter="\t")
                                for (pred, label) in zip(preds, labels):
                                    if pred == True and label == True:
                                        tp_writer.writerow(line)
                                    elif pred == True and label == False:
                                        fp_writer.writerow(line)
                                    elif pred == False and label == False:
                                        tn_writer.writerow(line)
                                    else:
                                        fn_writer.writerow(line)

            if i % 1000 == 0:
                print("processed", i, "examples ...")



    #acc = corr / total
    #prec = true_pos / (true_pos + false_pos)
    #rec = true_pos / (true_pos + false_neg)
    acc = metrics.accuracy_score(labels, preds)
    f1 = metrics.f1_score(labels, preds)
    prec = metrics.precision_score(labels, preds)
    rec = metrics.recall_score(labels, preds)
    conf_mat = metrics.confusion_matrix(labels, preds)

    acc_unk_obj = metrics.accuracy_score(labels_unk_obj, preds_unk_obj)
    f1_unk_obj = metrics.f1_score(labels_unk_obj, preds_unk_obj)
    prec_unk_obj = metrics.precision_score(labels_unk_obj, preds_unk_obj)
    rec_unk_obj = metrics.recall_score(labels_unk_obj, preds_unk_obj)
    conf_mat_unk_obj = metrics.confusion_matrix(labels_unk_obj, preds_unk_obj)

    with open(log_path, "w") as log_file:
        log_file.write("Vector sums with known objects as comparison:\n")
        #log_file.write(str(hyp_params) + "\n")
        #log_file.write("parameter file: " + model_path + "\n")
        log_file.write("stopwords: " + str(stopwords != []) + "\n")
        log_file.write("vector_file: " + vector_path + "\n")
        log_file.write("vocabulary size: " + str(len(vectors)) + "\n")
        log_file.write("cutoff: " + str(cutoff) + "\n")
        log_file.write("evaluated on: " + dev_path + "\n")
        log_file.write("accuracy: " + str(acc) + "\n")
        log_file.write("f1: " + str(f1) + "\n")
        log_file.write("precision: " + str(prec) + "\n")
        log_file.write("recall: " + str(rec) + "\n")
        log_file.write("confusion matrix: " + str(conf_mat) + "\n")
        log_file.write("\n\nresults for objects not seen during training:\n")
        log_file.write("accuracy: " + str(acc_unk_obj) + "\n")
        log_file.write("f1: " + str(f1_unk_obj) + "\n")
        log_file.write("precision: " + str(prec_unk_obj) + "\n")
        log_file.write("recall: " + str(rec_unk_obj) + "\n")
        log_file.write("confusion matrix: " + str(conf_mat_unk_obj) + "\n")
            
        

        
stopwords = ["", "(", ")", "a", "about", "an", "and", "are", "around", "as", "at",
    "away", "be", "become", "became", "been", "being", "by", "did", "do",
    "does", "during", "each", "for", "from", "get", "have", "has", "had", "he",
    "her", "his", "how", "i", "if", "in", "is", "it", "its", "made", "make",
    "many", "most", "not", "of", "on", "or", "s", "she", "some", "that", "the",
    "their", "there", "this", "these", "those", "to", "under", "was", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your"]


#smaller_set("../../../data/bert_classify_thereis_5caps_seed0/dev.tsv", 20000)

vectors = load_vectors("../../../data/glove.6B.300d.txt")
objects = load_objects("../../../data/bert_classify_thereis_5caps_seed0/train.tsv")
print(len(objects))
print("read vectors")


for cutoff in [4000, 5000, 6000]:
    evaluate_vector_addition(vectors, 
                             objects,
                             #"../../../data/bert_classify_thereis_5caps_seed0/dev.tsv",
                             "../../../data/bert_classify_thereis_5caps_seed0/dev_20000.tsv",
                             #"../../../../data/entailment_data_analysis/obj_in_caption/score/obj_no_cap_equal_TF/dev.tsv",
                             "../../../logs/vector_sum/w2v/" + str(cutoff) + ".log",
                             #"../../../logs/vector_sum/full/" + str(cutoff) + ".log",
                             #"/home/users/jseltmann/Misc/debug/vector_add.log",
                             cutoff=cutoff,
                             vector_path="../../../data/w2v_from_train_ignore_False.txt",
                             stopwords=stopwords)
