#!/usr/bin/env python
# coding: utf-8

# In[173]:


import numpy as np
import csv
from sklearn import metrics

# -----------------Scores of Finetuned BERT baseline---------------------#


# Path adapt

# path_not_seen = '/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/cap_not_seen/dev.tsv'
# no = 'Object_not_in_caption'
# path_seen = '/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/cap_seen/dev.tsv'
# yes = "Object_in_caption"

eq = 'equally_examples'
path_equal_examples = '/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/score/dev.tsv'


def scores(path,dataset_type):
    labels = []
    with open(path) as f:
        s_reader = csv.reader(f,delimiter = '\t', escapechar="\\")
        for line in s_reader:
            words = line[3].split()
            labels.append(words)
            
        flat_labels = []
        for items in labels:
            for sub in items:
                flat_labels.append(sub)
        flat_labels.pop(0)
        flat_labels_bin =[]
        
        
        for items in flat_labels:
            if items == "True":
                flat_labels_bin.append(0)
            else:
                flat_labels_bin.append(1)
        y_true = np.array(flat_labels_bin)
        
        print("F1 score :",dataset_type, ":", metrics.f1_score(y_true,y_pred))
        print("Recall :",dataset_type , ":",metrics.recall_score(y_true,y_pred))
        print("Precision: ",dataset_type , ":", metrics.precision_score(y_true,y_pred))
        print("Accuracy: ",dataset_type , ":", metrics.accuracy_score(y_true,y_pred))
        #print("Accuracy: ",dataset_type , ":", metrics.confusion_matrix(y_true,y_pred))

        
        
#----------------Uncomment below one line when using scores(path_equal_examples,equal_shuffled)---------- #
y_pred = np.load("/data/entailment_data_analysis/bert/preds_equal_examples.npy") 
scores(path_equal_examples,eq)


#----------------Uncomment below one line when using scores(path_not_seen,no)---------- #
# y_pred = np.load("/data/entailment_data_analysis/bert/preds.npy_not_seen")    
# scores(path_equal_examples,no)


# -----------------Uncomment below one line when using #scores(path_seen,yes)---------- #

#y_pred = np.load("/data/entailment_data_analysis/bert/preds.npy_seen")
#scores(path_seen,yes)
    
    ##########------------------Results --------------###############
 
"""
F1 score : equally_examples : 0.7757595772787319
Recall : equally_examples : 0.730183400683867
Precision:  equally_examples : 0.8274040154984149
Accuracy:  equally_examples : 0.8807097680955728
        
        
F1 score : Object_in_caption : 0.30123456790123454
Recall : Object_in_caption : 0.17941176470588235
Precision:  Object_in_caption : 0.9384615384615385
Accuracy:  Object_in_caption : 0.9502723598664559
        
        
F1 score : Object_not_in_caption : 0.8108385248031691
Recall : Object_not_in_caption : 0.7979430751579326
Precision:  Object_not_in_caption : 0.8241576242147345
Accuracy:  Object_not_in_caption : 0.8071213217127265

"""
        


# F1 score : Object_in_caption : 0.30123456790123454
# Recall : Object_in_caption : 0.17941176470588235
# Precision:  Object_in_caption : 0.9384615384615385
# Accuracy:  Object_in_caption : 0.9502723598664559
#         
# F1 score : Object_not_in_caption : 0.8108385248031691
# Recall : Object_not_in_caption : 0.7979430751579326
# Precision:  Object_not_in_caption : 0.8241576242147345
# Accuracy:  Object_not_in_caption : 0.8071213217127265
#         
#         
