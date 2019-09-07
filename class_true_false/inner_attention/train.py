import torch
import torch.nn.utils.rnn as rnn
import nltk
import csv
import pickle

from model import InnerAttLSTM

MAX_LEN = 25
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

INPUT_SIZE = 1
HIDDEN_SIZE = 25#100
with open("../../../data/bert_classify_thereis_5caps_seed0/word_inds.pkl", "rb") as word_ind_file:
    word2num, _ = pickle.load(word_ind_file)
    PAD_INDEX = word2num["<PAD>"]
NUM_WORDS = len(word2num)
EMB_SIZE=300

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
        train_reader = csv.reader(train_file, delimiter='\t', quotechar=None, escapechar="\\")
        for i, line in enumerate(train_reader):
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

            #inputs.append((cap_inds, obj_inds))
            
            if label == "True":
                labels.append(torch.tensor([0.0,1.0]))
            elif label == "False":
                labels.append(torch.tensor([1.0,0.0]))
            else:
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

    #return inputs_batched, labels_batched
    return caps_batched, objs_batched, labels_batched


#caps, objs, labels = load_data("../../data/bert_classify_thereis_5caps_seed0/dev.tsv",
#                           "../../data/bert_classify_thereis_5caps_seed0/word_inds.pkl",
#                           batch_size=BATCH_SIZE)
#with open("../../data/bert_classify_thereis_5caps_seed0/lstm_preprocessed_dev.pkl", "wb") as processed_file:
#    pickle.dump((caps, objs, labels), processed_file)
with open("../../../data/bert_classify_thereis_5caps_seed0/lstm_preprocessed_train.pkl", "rb") as processed_file:
    caps, objs, labels = pickle.load(processed_file)

num_batches = len(caps)

print("loaded data")

model = InnerAttLSTM(NUM_WORDS, EMB_SIZE, HIDDEN_SIZE, BATCH_SIZE, MAX_LEN)

loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):

    i = 0
    for (cap_batch, obj_batch), label_batch in list(zip(zip(caps, objs),labels)):
        cap_batch.sort(key=len, reverse=True)
        cap_batch = rnn.pack_sequence(cap_batch)#, padding_value=PAD_INDEX)
        cap_batch, _ = rnn.pad_packed_sequence(cap_batch, padding_value=PAD_INDEX, total_length=MAX_LEN)
        cap_batch = cap_batch.unsqueeze(2)

        obj_batch.sort(key=len, reverse=True)
        obj_batch = rnn.pack_sequence(obj_batch)
        obj_batch, _ = rnn.pad_packed_sequence(obj_batch, padding_value=PAD_INDEX)
        obj_batch = obj_batch.unsqueeze(2)

        label_batch = torch.stack(label_batch)

        model.zero_grad()
        model.reset_state()

        preds = model(cap_batch, obj_batch)
        loss = loss_fn(preds, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5000 == 0:
            print("epoch:", epoch, "batch:", i, "out of", num_batches)
        i += 1

torch.save(model.state_dict(), "../../../logs/base_lstm_classification/models/activations/inner_att.pt")
