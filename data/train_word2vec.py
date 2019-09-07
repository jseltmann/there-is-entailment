import gensim
import csv
import nltk


def train_w2v(csv_path, model_path, word_order=[]):
    """
    Train word2vec model based on the training set of the task.

    Parameters
    ----------
    csv_path : str
        Path to csv file containing the train set.
    model_path : str
        Path to save the trained model to.
    word_order : [str]
        List of words giving the order in which to save them (e.g. according to frequency).
        (A bit of a hack, since it is also possible to sort the vocabulary with gensim, 
        but I'm not sure how.)
    """

    train_examples = []

    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter='\t', escapechar="\\")
        for i, line in enumerate(reader):
            if i == 0:
                continue
            cap = nltk.word_tokenize(line[1].lower())
            obj = nltk.word_tokenize(line[2].lower())
            if line[3] == "True":
                example = cap + obj
                train_examples.append(example)
            else: # if there is no entailment, then the meanings of the words in the captions and in the object are probably not related
                train_examples.append(cap)
                train_examples.append(obj)
 
    print("loaded train data")

    print(len(train_examples))
    model = gensim.models.Word2Vec(train_examples, min_count=3, size=300, workers=2, window=5, iter=3, sg=0)
    print("trained model")

    words_to_save = set(model.wv.vocab.keys())

    with open(model_path, "w") as model_file:
        for word in word_order:
            if word in words_to_save:
                vec = model.wv.get_vector(word)
                vec_str = " ".join([str(entry) for entry in vec])
                line = word + " " + vec_str + "\n"
                model_file.write(line)
                words_to_save.remove(word)
        for word in words_to_save:
            vec = model.wv.get_vector(word)
            vec_str = " ".join([str(entry) for entry in vec])
            line = word + " " + vec_str + "\n"
            model_file.write(line)


word_order = []
with open("/home/users/jseltmann/there-is-entailment/data/glove.6B.300d.txt") as glove_file:
    for i, line in enumerate(glove_file):
        word = line.split()[0]
        word_order.append(word)
print("read vocabulary from glove file")

train_w2v("/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/train.tsv",
          "/home/users/jseltmann/there-is-entailment/data/w2v_from_train_ignore_False.txt",
          #"/home/users/jseltmann/Misc/debug/wv2_test.txt",
          word_order=word_order)
