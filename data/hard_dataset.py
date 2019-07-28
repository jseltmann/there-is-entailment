#!/usr/bin/env python
# coding: utf-8

# In[88]:



###########----------------Harder Dataset----------------------###############
##########---Check if the Object is mentioned in the caption---########


# File path variable
# needs to be adapted
path_train =  '../there-is-entailment/data/bert_classify_thereis_5caps_seed0/train.tsv'
path_test = '../there-is-entailment/data/bert_classify_thereis_5caps_seed0/test.tsv'
path_dev = '../there-is-entailment/data/bert_classify_thereis_5caps_seed0/dev.tsv'

# Save location 
# Files must be renamed into dev.tsv,train.tsv,test.tsv separately for caption_has_oject and caption_has_no_object
#and moved to the folder to read
save_train = '/data/entailment_data_analysis/obj_in_caption/train'
save_test = '/data/entailment_data_analysis/obj_in_caption/test'
save_dev = '/data/entailment_data_analysis/obj_in_caption/dev'


def WordInCaption(path,save):
    with open(path) as f:
        s_reader = csv.reader(f,delimiter = '\t', escapechar="\\")
        with open(save + '_has_obj.tsv',"w") as f1:
            s1_writer = csv.writer(f1, delimiter='\t', quotechar=None, escapechar="\\")
            with open(save +'_has_no_obj.tsv',"w") as f2:
                s2_writer = csv.writer(f2,delimiter='\t', quotechar=None, escapechar="\\")
                
                s1_writer.writerow(["index", "caption", "object", "entailment"])
                
                for line in s_reader:
                    words = line[1].split()
                    if line[2] in words:                        
                        s1_writer.writerow(line)
                    else:
                        s2_writer.writerow(line)    

WordInCaption(path_train,save_train)
WordInCaption(path_test,save_test)
WordInCaption(path_dev,save_dev)



# Total examples in all sets

dev_has_no_obj = 140284 # without seed0 140215
dev_has_obj = 5708 #5744
test_has_no_obj = 140604 #without seed0 140485 
test_has_obj = 5385 #without seed0 5532
train_has_no_obj =  1122135 # without seed0 1122351
train_has_obj = 45802 #without seed0 45585

obj_in_caption = dev_has_obj + test_has_obj + train_has_obj

obj_NOT_in_caption = dev_has_no_obj + test_has_no_obj + train_has_no_obj

total_obj_in_caption = obj_in_caption / (obj_in_caption + obj_NOT_in_caption)

total_obj_NOT_in_caption = obj_NOT_in_caption / (obj_in_caption + obj_NOT_in_caption)

Total = obj_in_caption + obj_NOT_in_caption


print("\nTraining examples which HAD object Mentioned in the Caption : ",train_has_obj )
print("\nTraining examples which DO NOT have object Mentioned in the Caption : ",train_has_no_obj)
print("\nTraining Examples with Object in Caption % : ",train_has_obj/ (train_has_obj + train_has_no_obj))


print("\nTest examples which HAD object Mentioned in the Caption : ",test_has_obj)
print("\nTest examples which DO NOT have object Mentioned in the Caption : ",test_has_no_obj)
print("\nTest Examples with Object in Caption % : ",test_has_obj/ (test_has_obj + test_has_no_obj))


print("\nDev examples which HAD object Mentioned in the Caption : ",dev_has_obj)
print("\nDev examples which DO NOT have object Mentioned in the Caption : ",dev_has_no_obj)
print("\nDev Examples with Object in Caption % : ",dev_has_obj/ (dev_has_obj + dev_has_no_obj))

print("\nTotal dataset Examples with Object in the caption : ", total_obj_in_caption)
print("\nTotal dataset Examples where object is NOT in the caption :",total_obj_NOT_in_caption )
print("\nTotal Examples", Total)

#Training examples which HAD object Mentioned in the Caption :  45802
#Training examples which DO NOT have object Mentioned in the Caption :  1122135
#Training Examples with Object in Caption % :  0.03921615635089906
#Test examples which HAD object Mentioned in the Caption :  5385
#Test examples which DO NOT have object Mentioned in the Caption :  140604
#Test Examples with Object in Caption % :  0.03688634075170047
#Dev examples which HAD object Mentioned in the Caption :  5708
#Dev examples which DO NOT have object Mentioned in the Caption :  140284
#Dev Examples with Object in Caption % :  0.03909803276891884
#Total dataset Examples with Object in the caption :  0.03897136688498943
#Total dataset Examples where object is NOT in the caption : 0.9610286331150105
#Total Examples 1459918
