import pickle
import numpy as np
import csv
import nltk
import ast

# helper class to keep track of word-to-id mappings
class Vocabulary:
    def __init__(self):
        self.count = 0
        self.vocab = dict()
        self.reverse = dict()

    def put(self, word):
        if word not in self.vocab:
            self.vocab[word] = self.count
            self.reverse[self.count] = word
            self.count = self.count + 1
        return self.vocab[word]

    def lookup(self, id):
        return self.reverse[id]    

# reads data from file
# data will be a triple: (dict(features), objectId, entailment)
def load_data(path, words):
    print('Loading.. ', path)
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        train = [] 
        for row in reader:
            id = row[0] 
            if id == 'index' or len(row) != 4:
                continue
            sentence = row[1].split()
            entailment = ast.literal_eval(row[3])
            obj = 'obj_' + row[2].lower()
            curr = dict()
            objid = words.put(obj)
        
            for word in sentence:
                word = word.lower()
                wordid = words.put(word)
                curr[wordid] = 1
            curr[objid] = 1
            train.append( (curr, entailment, entailment) )
        print('Loaded ', len(train), ' rows')
        return train

# strips away rows with negative entailment
def to_nltk_featureset(data):
    return [(feats, id) for feats, id, entail in data]

# evaluates classifier against a testset. takes into account both positive as well as negative samples
def evaluate(words, classifier, data):
    acc = []
    for case in data:
        entailment = case[2]

        #probDist = classifier.prob_classify(case[0])
        #probs = [(probDist.prob(sample), sample) for sample in probDist.samples()]
        #probs.sort(reverse=True)
        #predicted = [objects.lookup(w) for p, w in probs[:10]]

        #if (entailment and correct in predicted) or ((not entailment) and correct not in predicted):
        #    acc.append(1)
        #else:
        #    acc.append(0)
        acc.append(entailment == classifier.classify(case[0]))
    return sum(acc) / float(len(acc))

# trains a NaiveBayes classifier and evaluates it
def train_and_evaluate():
    words = Vocabulary()
    train  = load_data("/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps/dev.tsv", words)
    test = load_data("/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps/test.tsv", words)

    print('Vocabulary:', len(words.vocab))

    print('Training on data..')
    classifier = nltk.classify.NaiveBayesClassifier.train(to_nltk_featureset(train))

    print('Train Accuracy:', evaluate(words, classifier, train))
    print('Test  Accuracy:', evaluate(words, classifier, test))

train_and_evaluate()
