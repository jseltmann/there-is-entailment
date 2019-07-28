#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv

# File path variable needs to be adapted

path_train =  '../there-is-entailment/data/bert_classify_thereis_5caps_seed0/train.tsv'
path_test = '../there-is-entailment/data/bert_classify_thereis_5caps_seed0/test.tsv'
path_dev = '../there-is-entailment/data/bert_classify_thereis_5caps_seed0/dev.tsv'

# Needs to be adapted

save_train = '../data/entailment_data_analysis/obj_in_caption/train'
save_test = '../data/entailment_data_analysis/obj_in_caption/test'
save_dev = '../data/entailment_data_analysis/obj_in_caption/dev'

def WordInCaption(path,save):
    with open(path) as f:
        with open(save + '_has_obj.tsv',"w") as f1:
            with open(save +'_has_no_obj.tsv',"w") as f2:
                s_reader = csv.reader(f,delimiter = '\t', escapechar="\\")
                for line in s_reader:
                    words = line[1].split()
                    #print(words)
                    new_line = line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + line[3]  + "\n"
                    if line[2] in words:                        
                        f1.write(new_line)
                    else:
                        f2.write(new_line)                                 

            
WordInCaption(path_train,save_train)
WordInCaption(path_test,save_test)
WordInCaption(path_dev,save_dev)


# In[1]:



# Total examples in all sets

dev_false = 72700
dev_true = 73261
test_false = 72705
test_true = 73311
train_false  = 581600
train_true = 586335

only_false_ex = dev_false + test_false + train_false
only_true_ex = dev_true + test_true + train_true

Total_false_ex = only_false_ex / (only_false_ex+only_true_ex)
Total_true_ex = only_true_ex / (only_false_ex+only_true_ex)

Total = only_false_ex+only_true_ex

print("Examples where object is seen in image",Total_true_ex)
print("Examples where object is NOT seen in the image",Total_false_ex)
print("Total examples",Total)


#Examples where object is seen in image 0.5020213547117909
#Examples where object is NOT seen in the image 0.4979786452882092
#Total examples 1459912


# In[ ]:




