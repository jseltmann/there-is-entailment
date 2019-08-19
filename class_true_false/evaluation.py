import torch
import pickle
import torch.nn.utils.rnn as rnn
import sklearn.metrics as metrics

from basic_model.model import BaseLSTM
from attention.model import AttLSTM
from reuse_hidden_state.model import ReuseStateLSTM
from attention_embedding.model import EmbAttLSTM
from embedding.model import EmbLSTM
from inner_attention.model import InnerAttLSTM
from activations.model import ActLSTM

with open("../../data/bert_classify_thereis_5caps_seed0/word_inds.pkl", "rb") as word_ind_file:
    word2num, _ = pickle.load(word_ind_file)
    PAD_INDEX = word2num["<PAD>"]
NUM_WORDS = len(word2num)
MAX_LEN = 25


def evaluate_lstm(hyp_params, model, model_path, dev_path, log_path):
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
    """

    #input_size, hidden_size, batch_size = hyp_params
    #model = BaseLSTM(input_size, hidden_size, batch_size)
    #model.load_state_dict(torch.load(model_path))
    #model.eval()

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

    for (cap_batch, obj_batch), label_batch in list(zip(zip(caps, objs), labels)):
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
        #for pred, label in zip(preds, label_batch):
        #    count += 1
        #    if count % 20000 == 0 or count == 100:
        #        print("processed", count, "examples ...")
        #    pred = torch.argmax(pred)
        #    label = torch.argmax(label)
        #    total += 1

        #    if pred == label:
        #        corr += 1
        #    if label == 1 and pred == 1:
        #        true_pos += 1
        #    if label == 0 and pred == 1:
        #        false_pos += 1
        #    if label == 0 and pred == 0:
        #        true_neg += 1
        #    if label == 1 and pred == 0:
        #        false_neg += 1
    
    #acc = corr / total
    #prec = true_pos / (true_pos + false_pos)
    #rec = true_pos / (true_pos + false_neg)
    #neg_prec = true_neg / (true_neg + true_pos)
    #neg_rec = true_neg / (true_neg + false_neg)

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



#dev_path = "/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/cap_not_seen/lstm_preprocessed_dev.pkl"
dev_path = "/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/lstm_preprocessed_dev.pkl"
#log_path = "../../logs/base_lstm_classification/eval_results/hard_eval/"
log_path = "../../logs/base_lstm_classification/eval_results/easy_eval/"


#model = BaseLSTM(1, 25, 1)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/base_2019-08-02-2lin.pt"))
#model.eval()
#evaluate_lstm([1,25,1], 
#              model,
#              "../../logs/base_lstm_classification/models/base_2019-08-02-2lin.pt",
#              dev_path,
#              log_path + "base.log")
#print("evaluated BaseLSTM")
#
#
#model = EmbAttLSTM(NUM_WORDS, 300, 25, 1, 25)
#model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/emb_att_2019-08-06.pt"))
#model.eval()
#evaluate_lstm([NUM_WORDS, 300, 25, 1, 1], 
#              model,
#              "../../logs/base_lstm_classification/models/emb_att_2019-08-06.pt",
#              dev_path,
#              log_path + "emb_att.log")
#print("evaluated EmbAttLSTM")
#
#
model = ReuseStateLSTM(NUM_WORDS, 300, 25, 1)
model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/trans_state_model_2019-08-08.pt"))
model.eval()
evaluate_lstm([NUM_WORDS, 300, 25, 1], 
              model,
              "../../logs/base_lstm_classification/models/trans_state_model_2019-08-08.pt",
              dev_path,
              log_path + "reuse_state.log")
print("evaluated ReuseStateLSTM")


model = InnerAttLSTM(NUM_WORDS, 300, 25, 1, 25)
model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/inner_att_2019-08-15.pt"))
model.eval()
evaluate_lstm([NUM_WORDS, 300, 25, 1, 25], 
              model,
              "../../logs/base_lstm_classification/models/inner_att_2019-08-15.pt",
              dev_path,
              log_path + "inner_att.log")
print("evaluated InnerAttLSTM")


model = EmbLSTM(NUM_WORDS, 300, 25, 1)
model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/emb_model_2019-08-05.pt"))
model.eval()
evaluate_lstm([NUM_WORDS, 300, 25, 1, 25], 
              model,
              "../../logs/base_lstm_classification/models/emb_model_2019-08-05.pt",
              dev_path,
              log_path + "emb.log")
print("evaluated EmbLSTM")


model = AttLSTM(1, 25, 1, MAX_LEN)
model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/att_mean_2019-08-07.pt"))
model.eval()
evaluate_lstm([1, 25, 1, MAX_LEN], 
              model,
              "../../logs/base_lstm_classification/models/att_mean_2019-08-07.pt",
              dev_path,
              log_path + "att.log")
print("evaluated AttLSTM")


model = LSTM(1, 25, 1)
model.load_state_dict(torch.load("../../logs/base_lstm_classification/models/base_2019-08-08_activations.pt"))
model.eval()
evaluate_lstm([1, 25, 1], 
               model,
               "../../logs/base_lstm_classification/models/base_2019-08-08_activations.pt",
               dev_path,
               log_path + "activations.log")
print("evaluated ActLSTM")
