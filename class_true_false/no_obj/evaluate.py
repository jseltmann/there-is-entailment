import torch
import pickle
import torch.nn.utils.rnn as rnn

from model import EmbLSTM

#PAD_INDEX = 1889
EMB_DIM=300
INPUT_SIZE=1
HIDDEN_SIZE=25
BATCH_SIZE=1
with open("../../../data/bert_classify_thereis_5caps_seed0/word_inds.pkl", "rb") as word_ind_file:
    word2num, _ = pickle.load(word_ind_file)
    PAD_INDEX = word2num["<PAD>"]
NUM_WORDS = len(word2num)

def evaluate_base_lstm(hyp_params, model_path, dev_path, log_path):
    """
    Evaluate LSTM model by calculating the accuracy over a test set.

    Parameters
    ----------
    hyp_params : tuple(int)
        4-tuple containing the size of the vocabulary, the embedding size, the hidden size, and the batch size of the model.
    model_path : str
        Path of file containing the weights of the model.
    dev_path : str
        Path of file containing the set on which to evaluate the model.
    log_path : str
        Path to write the results to.
    """

    num_words, emb_dim, hidden_size, batch_size = hyp_params
    model = EmbLSTM(num_words, emb_dim, hidden_size, batch_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(dev_path, "rb") as f:
        caps, objs, labels = pickle.load(f)

    corr = 0
    total = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    count = 0


    for (cap_batch, obj_batch), label_batch in list(zip(zip(caps, objs), labels)):
        cap_batch.sort(key=len, reverse=True)
        cap_batch = rnn.pack_sequence(cap_batch)
        cap_batch, _ = rnn.pad_packed_sequence(cap_batch, padding_value=PAD_INDEX)
        cap_batch = cap_batch.unsqueeze(2)
        obj_batch.sort(key=len, reverse=True)
        obj_batch = rnn.pack_sequence(obj_batch)
        obj_batch, _ = rnn.pad_packed_sequence(obj_batch, padding_value=PAD_INDEX)
        obj_batch = obj_batch.unsqueeze(2)
        
        preds = model(cap_batch, obj_batch)

        for pred, label in zip(preds, label_batch):
            count += 1
            if count % 20000 == 0 or count == 100:
                print("processed", count, "examples ...")
            pred = torch.argmax(pred)
            label = torch.argmax(label)
            total += 1

            if pred == label:
                corr += 1
            if label == 1 and pred == 1:
                true_pos += 1
            if label == 0 and pred == 1:
                false_pos += 1
            if label == 0 and pred == 0:
                true_neg += 1
            if label == 1 and pred == 0:
                false_neg += 1
    
    acc = corr / total
    prec = true_pos / (true_pos + true_neg)
    rec = true_pos / (true_pos + false_pos)
    neg_prec = true_neg / (true_neg + true_pos)
    neg_rec = true_neg / (true_neg + false_neg)

    with open(log_path, "w") as log_file:
        log_file.write("EmbModel\n")
        log_file.write(str(hyp_params) + "\n")
        log_file.write("parameter file: " + model_path + "\n")
        log_file.write("evaluated on: " + dev_path + "\n")
        log_file.write("accuracy: " + str(acc) + "\n")
        log_file.write("precision: " + str(prec) + "\n")
        log_file.write("recall: " + str(rec) + "\n")
        log_file.write("neg precision: " + str(neg_prec) + "\n")
        log_file.write("neg recall: " + str(neg_rec) + "\n")

        
evaluate_base_lstm([NUM_WORDS, EMB_DIM, HIDDEN_SIZE, BATCH_SIZE],
                   "../../../logs/base_lstm_classification/models/emb_model_2019-08-05.pt",
                   "../../../data/bert_classify_thereis_5caps_seed0/lstm_preprocessed_dev.pkl", 
                   "../../../logs/base_lstm_classification/eval_results/2019-08-16_emb_easy_dev.log")
