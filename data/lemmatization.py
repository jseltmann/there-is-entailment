#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

path_train =  '/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/train.tsv'
path_test = '/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/test.tsv'
path_dev = '/home/users/jseltmann/there-is-entailment/data/bert_classify_thereis_5caps_seed0/dev.tsv'

save_train = '/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/train'
save_test = '/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/test'
save_dev = '/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/dev'


def LemmaHardSet(path,save):
    with open(path_dev) as f:
        s_reader = csv.reader(f,delimiter = '\t', escapechar="\\")
        with open(save_dev + '_has_obj.tsv',"w") as f1:
            s_writer1 = csv.writer(f1,delimiter = '\t', escapechar="\\")
            with open(save_dev +'_has_no_obj.tsv',"w") as f2:
                s_writer2 = csv.writer(f2,delimiter = '\t', escapechar="\\")
                nlp = spacy.load('en', disable=['parser', 'ner'])
                for line in s_reader:
                    doc1 = nlp(line[1])
                    doc2 = nlp(line[2])
                    sen1 = " ".join([token.lemma_ for token in doc1])
                    sen2 = " ".join([token.lemma_ for token in doc2])
                    if sen2 in sen1:
                        #print(line)  
                        s_writer1.writerow(line)
                    else:
                        s_writer2.writerow(line)
                    #print(line)

LemmaHardSet(path_train,save_train)
#LemmaHardSet(path_dev,save_dev)
LemmaHardSet(path_test,save_test)


# In[ ]:




