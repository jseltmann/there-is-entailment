import torch
import pickle
import torch.nn.utils.rnn as rnn
import sklearn.metrics as metrics
import csv
import os, os.path
import warnings
import numpy as np
import collections as col

from basic_model.model import BaseLSTM
from attention.model import AttLSTM
from reuse_hidden_state.model import ReuseStateLSTM
from attention_embedding.model import EmbAttLSTM
from embedding.model import EmbLSTM
from inner_attention.model import InnerAttLSTM
from activations.model import ActLSTM
from att_emb_act.model import EmbAttActLSTM
from stacked_lstm.model import StackedLSTM
from no_cap.model import NoCapLSTM
from no_obj.model import NoObjLSTM
from transformer.model import WithTransf
from preload_emb.model import PreloadEmbLSTM
from transformer_comb_inp.model import WithTransfCombInp


with open("../../data/bert_classify_thereis_5caps_seed0/word_inds.pkl", "rb") as word_ind_file:
    word2num, _ = pickle.load(word_ind_file)
    PAD_INDEX = word2num["<PAD>"]
NUM_WORDS = len(word2num)
MAX_LEN = 25


def compare_for_lengths(log_dir, length_list, log_name, stopwords=[]):
    """
    Compare how many examples were classified correctly split up by caption lengths.

    Parameters
    ----------
    log_dir : str
        Directory containing the result files of the classification
        and where the results of this function are saved to.
    length_list : str
        List containing boundaries of length bins for the analysis.
    log_name : str
        Filename to write the results to.
    stopwords : [str]
        List of words to ignore in the length count.
    """

    pos_dict = dict()
    length_list = sorted(length_list)
    pos = 0
    max_len = max(length_list)
    for i in range(max_len+1):
        pos_dict[i] = pos
        if i in length_list:
            pos += 1

    true_pos_list = [0] * len(length_list)
    with open(log_dir + "/true_pos.log") as log_file:
        reader = csv.reader(log_file, delimiter="\t", escapechar="\\")
        for line in reader:
            cap = line[1].split()
            stopped = []
            for word in cap:
                if word.lower() not in stopwords:
                    stopped.append(word)
            cap = stopped
            length = len(cap)
            if length > max_len:
                length = max_len
            pos = pos_dict[length]
            true_pos_list[pos] += 1

    false_pos_list = [0] * len(length_list)
    with open(log_dir + "/false_pos.log") as log_file:
        reader = csv.reader(log_file, delimiter="\t", escapechar="\\")
        for line in reader:
            cap = line[1].split()
            stopped = []
            for word in cap:
                if word.lower() not in stopwords:
                    stopped.append(word)
            cap = stopped
            length = len(cap)
            if length > max_len:
                length = max_len
            pos = pos_dict[length]
            false_pos_list[pos] += 1

    true_neg_list = [0] * len(length_list)
    with open(log_dir + "/true_neg.log") as log_file:
        reader = csv.reader(log_file, delimiter="\t", escapechar="\\")
        for line in reader:
            cap = line[1].split()
            stopped = []
            for word in cap:
                if word.lower() not in stopwords:
                    stopped.append(word)
            cap = stopped
            length = len(cap)
            if length > max_len:
                length = max_len
            pos = pos_dict[length]
            true_neg_list[pos] += 1
            
    false_neg_list = [0] * len(length_list)
    with open(log_dir + "/false_neg.log") as log_file:
        reader = csv.reader(log_file, delimiter="\t", escapechar="\\")
        for line in reader:
            cap = line[1].split()
            stopped = []
            for word in cap:
                if word.lower() not in stopwords:
                    stopped.append(word)
            cap = stopped
            length = len(cap)
            if length > max_len:
                length = max_len
            pos = pos_dict[length]
            false_neg_list[pos] += 1

    tparr = np.array(true_pos_list)
    fnarr = np.array(false_neg_list)
    tnarr = np.array(true_neg_list)
    fparr = np.array(false_pos_list)

    acc_arr = (tparr + tnarr) / (tparr + tnarr + fnarr + fparr)
    prec_arr = tparr / (tparr + fparr)
    rec_arr = tparr / (tparr + fnarr)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr)

    acc_arr = [round(x,3) for x in acc_arr]
    prec_arr = [round(x,3) for x in prec_arr]
    rec_arr = [round(x,3) for x in rec_arr]
    f1_arr = [round(x,3) for x in f1_arr]

    metrics = zip(acc_arr, f1_arr, prec_arr, f1_arr)
    conf_matrices = zip(tnarr, fparr, fnarr, tparr)

    with open(log_dir + log_name, "w") as log_file:
        log_file.write("length\ttotal\tacc\tf1\tprec\trec\t(tn,tp,fn,tp)\n")
        for l, (acc,f1,prec,rec), conf_mat in zip(length_list, metrics, conf_matrices):
            conf_mat = list(conf_mat)
            total = sum(conf_mat)
            strs = [str(v) for v in [l,total,acc,f1,prec,rec,conf_mat]]
            line = "\t".join(strs) + "\n"
            log_file.write(line)


def get_split_ratios(cont_dir, log_filename):
    """
    For the split pentuples found in same_class_for_cap,
    find the true/false ratios.

    Parameters
    ----------
    cont_dir : str
        Directory containing the files produced by same_class_for_cap().
    log_filename : str
        File to write the results to.
    """

    count_dict = dict()
    for filename, label in [("/false_examples_cont.log", False), ("/true_examples_cont.log", True)]:
        count_dict[label] = col.defaultdict(int)
        split = False
        pentuples = []
        with open(cont_dir + filename) as cont_file:
            lines = cont_file.readlines()
            while lines != []:
                if lines[0] == "split:\n":
                    split = True
                    lines = lines[1:]
                    continue
                if not split:
                    lines = lines[1:]
                    continue
                pentuple = lines[:5]
                preds = []
                for l in pentuple:
                    l = l.split("\t")
                    if l[1] == "True\n":
                        pred = True
                    elif l[1] == "False\n":
                        pred = False
                    preds.append(pred)
                lines = lines[6:]
                tc = len([x for x in pred if x])
                fc = len([x for x in pred if not c])
                count_dict[label][(tc,fc)] += 1

    with open(log_filename, "w") as log_file:
        log_file.write("(true, false)\n\n")
        for label in count_dict.keys():
            log_file.write(str(label) + " examples:\n")
            items = count_dict[label].items()
            items = sorted(items, key=lambda x: x[0][0])
            for item in items:
                log_file.write(str(item[0]) + "\t" + str(item[1]) + "\n")
            log_file.write("\n")

           
def same_class_for_cap(log_dir):
    """
    For the result files of the classification, find out if the different captions for the same image were also classified the same way.

    Parameters
    ----------
    log_dir : str
        Directory where true_pos.log etc. are saved. The output files of this function are also saved there.
    """

    with open(log_dir + "/cont_captions.log", "w") as log_file:
        compl_corr = []
        compl_wrong = []
        split = []
        for filename, pred in [("/true_pos.log", True), ("/false_neg.log", False)]:
            num_compl_corr = 0
            num_compl_wrong = 0
            num_split = 0
            with open(log_dir + filename) as tsv_file:
                reader = csv.reader(tsv_file, delimiter="\t", escapechar="\\")
                prev_ids = []
                prev_lines = []
                for i, line in enumerate(reader):
                    ii = int(line[0])
                    prev_ids.append(ii)
                    prev_lines.append((line, pred))
                    if len(prev_ids) == 5:
                        if prev_ids[4] - prev_ids[0] == 4 and prev_ids[0] % 5 == 0:
                            if pred:
                                compl_corr.append(prev_lines)
                                num_compl_corr += 5
                            else:
                                compl_wrong.append(prev_lines)
                                num_compl_wrong += 5
                            prev_lines = []
                            prev_ids = []
                        else:
                            split.append(prev_lines[0])
                            prev_lines = prev_lines[1:]
                            prev_ids = prev_ids[1:]
                            num_split += 1
                if len(prev_ids) == 5:
                    if prev_ids[4] - prev_ids[0] == 4 and prev_ids[0] % 5 == 0:
                        if pred:
                            compl_corr += prev_lines
                            num_compl_corr += 5
                        else:
                            compl_wrong += prev_lines
                            num_compl_wrong += 5
                    else:
                        split += prev_lines
                        num_split += len(prev_lines)
                else:
                    split += prev_lines
                    num_split += len(prev_lines)
                with open(log_dir + filename[:-4] + "_cont_examples.log", "w") as log:
                    log.write("completely correct: " + str(num_compl_corr) + "\n")
                    log.write("completely wrong: " + str(num_compl_wrong) + "\n")
                    log.write("split: " + str(num_split) + "\n")
        with open(log_dir + "/true_examples_cont.log", "w") as examples_log:
            examples_log.write("###################\n")
            examples_log.write("completely correct:\n")
            for quintuple in compl_corr:
                for line, pred in quintuple:
                    examples_log.write(str(line) + "\t" + str(pred) + "\n")
                examples_log.write("\n")

            examples_log.write("\n\n\n\n###################\n")
            examples_log.write("completely false:\n")
            for quintuple in compl_wrong:
                for line, pred in quintuple:
                    examples_log.write(str(line) + "\t" + str(pred) + "\n")
                examples_log.write("\n")

            examples_log.write("\n\n\n\n###################\n")
            examples_log.write("split:\n")
            split = sorted(split, key=(lambda x: int(x[0][0])))
            for j, (line, pred) in enumerate(split):
                examples_log.write(str(line) + "\t" + str(pred) + "\n")
                if j % 5 == 4:
                    examples_log.write("\n")

        compl_corr = []
        compl_wrong = []
        split = []
        for filename, pred in [("/false_pos.log", True), ("/true_neg.log", False)]:
            num_compl_corr = 0
            num_compl_wrong = 0
            num_split = 0
            with open(log_dir + filename) as tsv_file:
                reader = csv.reader(tsv_file, delimiter="\t", escapechar="\\")
                prev_ids = []
                prev_lines = []
                for i, line in enumerate(reader):
                    ii = int(line[0])
                    prev_ids.append(ii)
                    prev_lines.append((line, pred))
                    if len(prev_ids) == 5:
                        if ii == 4:
                            print(prev_ids)
                        if prev_ids[4] - prev_ids[0] == 4 and prev_ids[0] % 5 == 0:
                            if not pred:
                                compl_corr.append(prev_lines)
                                num_compl_corr += 5
                            else:
                                compl_wrong.append(prev_lines)
                                num_compl_wrong += 5
                            prev_lines = []
                            prev_ids = []
                        else:
                            split.append(prev_lines[0])
                            prev_lines = prev_lines[1:]
                            prev_ids = prev_ids[1:]
                            num_split += 1
                if len(prev_ids) == 5:
                    if prev_ids[4] - prev_ids[0] == 4 and prev_ids[0] % 5 == 0:
                        if not pred:
                            compl_corr += prev_lines
                            num_compl_corr += 5
                        else:
                            compl_wrong += prev_lines
                            num_compl_wrong += 5
                        prev_lines = []
                split += prev_lines
                num_split += len(prev_lines)
                prev_lines = []
                with open(log_dir + filename[:-4] + "_cont_examples.log", "w") as log:
                    log.write("completely correct: " + str(num_compl_corr) + "\n")
                    log.write("completely wrong: " + str(num_compl_wrong) + "\n")
                    log.write("split: " + str(num_split) + "\n")
        with open(log_dir + "/false_examples_cont.log", "w") as examples_log:
            examples_log.write("###################\n")
            examples_log.write("completely correct:\n")
            for j, quintuple in enumerate(compl_corr):
                for line, pred in quintuple:
                    examples_log.write(str(line) + "\t" + str(pred) + "\n")
                examples_log.write("\n")

            examples_log.write("\n\n\n\n####################\n")
            examples_log.write("completely false:\n")
            for quintuple in compl_wrong:
                for line, pred in quintuple:
                    examples_log.write(str(line) + "\t" + str(pred) + "\n")
                examples_log.write("\n")

            examples_log.write("\n\n\n\n####################\n")
            examples_log.write("split:\n")
            split = sorted(split, key=(lambda x: int(x[0][0])))

            for j, (line, pred) in enumerate(split):
                examples_log.write(str(line) + "\t" + str(pred) + "\n")
                if j % 5 == 4:
                    examples_log.write("\n")

        



def evaluate_lstm(hyp_params, model, model_path, dev_path, log_path, tsv_path=None):
    """
    Evaluate LSTM model by calculating the accuracy over a test set.

    Parameters
    ----------
    hyp_params : tuple(int)
        3-tuple containing the input size, the hidden size, and the batch size of the model.
    model : nn.Module
        Model to be evaluated.
    model_path : str
        Path to the saved model weights
    dev_path : str
        Path of file containing the set on which to evaluate the model.
    log_path : str
        Path to write the results to.
    tsv_path : str
        Path of the file containing the original examples.
    """

    with open(dev_path, "rb") as f:
        caps, objs, labels = pickle.load(f)

    corr = 0
    total = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    count = 0

    preds = []
    label_list = []
    batches = 0
    
    tsv_lines = []
    if tsv_path:
        with open(tsv_path) as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t', escapechar="\\")
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                tsv_lines.append(line)

    i = 0
    for (cap_batch, obj_batch), label_batch in list(zip(zip(caps, objs), labels)):
        batches += 1
        cap_batch.sort(key=len, reverse=True)
        cap_batch = rnn.pack_sequence(cap_batch)
        cap_batch, _ = rnn.pad_packed_sequence(cap_batch, padding_value=PAD_INDEX, total_length=MAX_LEN)
        cap_batch = cap_batch.unsqueeze(2)
        obj_batch.sort(key=len, reverse=True)
        obj_batch = rnn.pack_sequence(obj_batch)
        obj_batch, _ = rnn.pad_packed_sequence(obj_batch, padding_value=PAD_INDEX, total_length=MAX_LEN)
        obj_batch = obj_batch.unsqueeze(2)
        
        curr_preds = model(cap_batch, obj_batch)

        preds += [torch.argmax(pred) for pred in curr_preds]
        label_list += [torch.argmax(label) for label in label_batch]

    if tsv_path:
        if len(log_path) > 4:
            log_dir = log_path[:-4]
        else:
            log_dir = log_path + "_dir"

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
                        for ((pred, label), line) in zip(zip(preds, label_list), tsv_lines):
                            if pred == 1 and label == 1:
                                tp_writer.writerow(line)
                            elif pred == 1 and label == 0:
                                fp_writer.writerow(line)
                            elif pred == 0 and label == 0:
                                tn_writer.writerow(line)
                            else:
                                fn_writer.writerow(line)

    acc = metrics.accuracy_score(label_list, preds)
    f1 = metrics.f1_score(label_list, preds)
    prec = metrics.precision_score(label_list, preds)
    rec = metrics.recall_score(label_list, preds)
    conf_mat = metrics.confusion_matrix(label_list, preds)

    with open(log_path, "w") as log_file:
        log_file.write(str(type(model)) + "\n")
        log_file.write(str(hyp_params) + "\n")
        log_file.write("parameter file: " + model_path + "\n")
        log_file.write("evaluated on: " + dev_path + "\n")
        log_file.write("accuracy: " + str(acc) + "\n")
        log_file.write("f1: " + str(f1) + "\n")
        log_file.write("precision: " + str(prec) + "\n")
        log_file.write("recall: " + str(rec) + "\n")
        log_file.write("confusion matrix: " + str(conf_mat) + "\n")



#dev_path = "/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/lstm_preprocessed_dev.pkl"
#dev_path = "/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/lstm_preprocessed_train.pkl"
dev_path = "/home/users/jseltmann/data/entailment_data_analysis/obj_in_caption/score/obj_no_cap_equal_TF/lstm_preprocessed_dev.pkl"

log_path = "../../logs/base_lstm_classification/eval_results/hard_eval/"
#log_path = "../../../Misc/debug/"
#log_path = "../../logs/base_lstm_classification/eval_results/easy_eval/"
#log_path = "../../logs/base_lstm_classification/eval_results/on_train/"

#tsv_path = "/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/score/dev.tsv"
#tsv_path = "/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/dev.tsv"
tsv_path = "/home/users/jseltmann/data/entailment_data_analysis/obj_in_caption/score/obj_no_cap_equal_TF/dev.tsv"
#tsv_path = "/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/train.tsv"


#model = BaseLSTM(1, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/base_25hidden.pt"))
#model.eval()
#evaluate_lstm([1,25,1], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/base_25hidden.pt",
#              dev_path,
#              log_path + "base.log",
#              tsv_path)
#print("evaluated BaseLSTM")
#
#
#model = EmbAttLSTM(NUM_WORDS, 300, 25, 64, MAX_LEN)
##model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/emb_att_2019-08-06.pt"))
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/emb_att.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 64, MAX_LEN], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/emb_att.pt",
#              dev_path,
#              log_path + "emb_att.log",
#              tsv_path)
#print("evaluated EmbAttLSTM")
#
#
#model = ReuseStateLSTM(NUM_WORDS, 300, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/trans_state.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/trans_state.pt",
#              dev_path,
#              log_path + "reuse_state.log",
#              tsv_path)
#print("evaluated ReuseStateLSTM")
#
#
#model = InnerAttLSTM(NUM_WORDS, 300, 25, 1, 25)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/inner_att.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1, 25], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/inner_att.pt",
#              dev_path,
#              log_path + "inner_att.log",
#              tsv_path)
#print("evaluated InnerAttLSTM")
#
#
#model = EmbLSTM(NUM_WORDS, 300, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/emb_model.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1, 25], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/emb_model.pt",
#              dev_path,
#              log_path + "emb.log",
#              tsv_path)
#print("evaluated EmbLSTM")
#
#
#model = AttLSTM(1, 25, 1, MAX_LEN)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/att.pt"))
#model.eval()
#evaluate_lstm([1, 25, 1, MAX_LEN], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/att.pt",
#              dev_path,
#              log_path + "att.log",
#              tsv_path)
#print("evaluated AttLSTM")
#
#
#model = StackedLSTM(NUM_WORDS, 300, 25, 1, MAX_LEN)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/stacked_lstm.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 300, 1, MAX_LEN], 
#               model,
#               "../../logs/base_lstm_classification/models/activations/stacked_lstm.pt",
#               dev_path,
#               log_path + "stacked_lstm.log",
#               tsv_path)
#print("evaluated StackedLSTM")
#
#
#model = NoCapLSTM(NUM_WORDS, 300, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/no_cap.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/no_cap.pt",
#              dev_path,
#              log_path + "no_cap.log",
#              tsv_path)
#print("evaluated NoCapLSTM")
#
#
#model = NoObjLSTM(NUM_WORDS, 300, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/no_object.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/no_object.pt",
#              dev_path,
#              log_path + "no_obj.log",
#              tsv_path)
#print("evaluated NoObjLSTM")
#
#
#model = WithTransf(NUM_WORDS, 320, MAX_LEN, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/withTransformer.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 320, MAX_LEN, 25, 1], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/withTransformer.pt",
#              dev_path,
#              log_path + "withTransf.log",
#              tsv_path)
#print("evaluated transformer")
#
#
#model = BaseLSTM(1, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/base_cross_ent_25hidden.pt"))
#model.eval()
#evaluate_lstm([1,25,1], 
#              model,
#              "../../logs/base_lstm_classification/models/activations/base_cross_ent_25hidden.pt",
#              dev_path,
#              log_path + "base_cross_ent.log",
#              tsv_path)
#print("evaluated BaseLSTM with crossentropy")
#
#
#VECTOR_PATH = "../../data/glove.6B.300d.txt"
#model = PreloadEmbLSTM(NUM_WORDS, 300, 25, 1, VECTOR_PATH, word2num)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/preload_emb_glove.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1, VECTOR_PATH],
#              model,
#              "../../logs/base_lstm_classification/models/activations/preload_emb_glove.pt",
#              dev_path,
#              log_path + "preload_new_glove.log",
#              tsv_path)
#print("evaluated PreloadEmbLSTM with GloVe")
#
#
#model = PreloadEmbLSTM(NUM_WORDS, 300, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/preload_emb_glove.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1, VECTOR_PATH],
#              model,
#              "../../logs/base_lstm_classification/models/activations/preload_emb_glove.pt",
#              dev_path,
#              log_path + "preload_trained_glove.log",
#              tsv_path)
#print("evaluated PreloadEmbLSTM with GloVe (without reloading)")
#
#
#model = WithTransfCombInp(NUM_WORDS, 320, MAX_LEN, 25, 64, PAD_INDEX)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/transf_comb_inp.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 320, MAX_LEN, 25, 1, PAD_INDEX],
#              model,
#              "../../logs/base_lstm_classification/models/activations/transf_comb_inp.pt",
#              dev_path,
#              log_path + "transf_comb_inp.log",
#              tsv_path)
#print("evaluated WithTransfCombInpLSTM")
#
#
#VECTOR_PATH = "../../data/w2v_from_train.txt"
#model = PreloadEmbLSTM(NUM_WORDS, 300, 25, 1, VECTOR_PATH, word2num)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/preload_emb_w2v_model.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1, VECTOR_PATH],
#              model,
#              "../../logs/base_lstm_classification/models/activations/preload_emb_w2v_model.pt",
#              dev_path,
#              log_path + "preload_new_w2v.log",
#              tsv_path)
#print("evaluated PreloadEmbLSTM with word2vec")
#
#
#model = PreloadEmbLSTM(NUM_WORDS, 300, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/activations/preload_emb_w2v_model.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1, VECTOR_PATH],
#              model,
#              "../../logs/base_lstm_classification/models/activations/preload_emb_w2v_model.pt",
#              dev_path,
#              log_path + "preload_trained_w2v.log",
#              tsv_path)
#print("evaluated PreloadEmbLSTM with word2vec (without reloading)")

easy_path = "/home/users/jseltmann/there-is-entailment/logs/base_lstm_classification/eval_results/easy_eval/"
hard_path = "/home/users/jseltmann/there-is-entailment/logs/base_lstm_classification/eval_results/hard_eval/"

#for cat in ["base", "att", "base_cross_ent", "emb", "emb_att", "inner_att", "no_cap",
#            "no_obj", "preload_new_glove", "preload_new_w2v", "reuse_state", "stacked_lstm"]:
for cat in ["preload_new_glove"]:
    #same_class_for_cap(easy_path + cat)
    same_class_for_cap(hard_path + cat)
    print(cat)


#length_list = [7,8,9,10,11,12,13,14,15,16]
#length_list_stopwords = [3,4,5,6,7,8,9,10]
#
#stopwords = ["", "(", ")", "a", "about", "an", "and", "are", "around", "as", "at",
#    "away", "be", "become", "became", "been", "being", "by", "did", "do",
#    "does", "during", "each", "for", "from", "get", "have", "has", "had", "he",
#    "her", "his", "how", "i", "if", "in", "is", "it", "its", "made", "make",
#    "many", "most", "not", "of", "on", "or", "s", "she", "some", "that", "the",
#    "their", "there", "this", "these", "those", "to", "under", "was", "were",
#    "what", "when", "where", "which", "who", "will", "with", "you", "your"]
#
#for cat in ["base", "att", "base_cross_ent", "emb", "emb_att", "inner_att", "no_cap",
#            "no_obj", "preload_new_glove", "preload_new_w2v", "reuse_state", "stacked_lstm"]:
#    #compare_for_lengths(easy_path + cat, length_list, "/length_anal.log")
#    #compare_for_lengths(hard_path + cat, length_list, "/length_anal.log")
#    compare_for_lengths(easy_path + cat, length_list_stopwords, 
#                        "/length_anal_stopwords.log", stopwords=stopwords)
#    compare_for_lengths(hard_path + cat, length_list_stopwords, 
#                        "/length_anal_wtopwords.log", stopwords=stopwords)
#    print(cat)

#for cat in ["base", "att", "base_cross_ent", "emb", "emb_att", "inner_att", "no_cap",
#            "no_obj", "preload_new_glove", "preload_new_w2v", "reuse_state", "stacked_lstm"]:
#    get_split_ratios(easy_path + cat, easy_path + cat + "/fractions.log")
#    get_split_ratios(hard_path + cat, hard_path + cat + "/fractions.log")
#    print(cat)
