import torch
import pickle
import torch.nn.utils.rnn as rnn
import numpy as np
import csv
from scipy.spatial.distance import cosine



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




def evaluate_vector_addition(vectors, dev_path, log_path, cutoff=40, vector_path=None):
    """
    Evaluate LSTM model by calculating the accuracy over a test set.

    Parameters
    ----------
    vectors : dict[str]
        Dict containing the vector for each word.
    dev_path : str
        Path of file containing the set on which to evaluate the model.
    log_path : str
        Path to write the results to.
    vector_path : str
        Path to file containing vectors.
    """

    vector_shape = list(vectors.values())[0].shape

    corr = 0
    total = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0

    with open(dev_path) as dev_file:
        dev_reader = csv.reader(dev_file, delimiter="\t", quotechar=None, escapechar="\\")
        for i, line in enumerate(dev_reader):
            if i == 0:
                continue
            try:
                _, cap, obj, label = line #line.split("\t")
            except Exception as e:
                print(line)
                7 / 0
            cap = cap.split()
            obj = obj.split()
            label = bool(label)
    
            cap_sum = np.zeros(vector_shape)
            for word in cap:
                word = word.lower()
                if word in vectors:
                    cap_sum += vectors[word]
    
            obj_sum = np.zeros(vector_shape)
            for word in obj:
                word = word.lower()
                if word in vectors:
                    obj_sum += vectors[word]
    
            obj_dist = cosine(cap_sum, obj_sum)
    
            #word_dists = []
            closer_count = 0
            for j, word in enumerate(vectors):
                curr_dist = cosine(cap_sum, vectors[word])
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
    
            if i % 20000 == 0 or i == 100:
                print("processed", i, "examples ...")

            total += 1
            if pred == label:
                corr += 1
            if label == True and pred == True:
                true_pos += 1
            if label == False and pred == True:
                false_pos += 1
            if label == False and pred == False:
                true_neg += 1
    
    acc = corr / total
    prec = true_pos / (true_pos + true_neg)
    rec = true_pos / (true_pos + false_pos)

    with open(log_path, "w") as log_file:
        log_file.write("Vector sums:\n")
        #log_file.write(str(hyp_params) + "\n")
        #log_file.write("parameter file: " + model_path + "\n")
        log_file.write("vector_file: " + vector_path + "\n")
        log_file.write("vocabulary size: " + str(len(vectors)) + "\n")
        log_file.write("cutoff: " + str(cutoff) + "\n")
        log_file.write("evaluated on: " + dev_path + "\n")
        log_file.write("accuracy: " + str(acc) + "\n")
        log_file.write("precision: " + str(prec) + "\n")
        log_file.write("recall: " + str(rec) + "\n")
            
        

        

#vectors = load_vectors("../../../data/glove.6B.300d.txt")
#print("read vectors")
#
#evaluate_vector_addition(vectors, 
#                         "../../../data/bert_classify_thereis_5caps_seed0/dev.tsv",
#                         "../../../logs/vector_sum/vector_sum_300d.log",
#                         cutoff=40,
#                         vector_path="../../../data/glove.6B.300d.txt")
