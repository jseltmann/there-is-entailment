#!/bin/bash

# This file first needs to have  harder dataset generated 
# It will use the dev set of the object IN/NOT in captions
# It will generate  file with equal number of  object IN/NOT in captions


tail -n +2 ../dev_has_obj.tsv > dev_set1 # 5711
tail -n +2 ../dev_has_no_obj.tsv > dev_set2 #
chmod 777 dev_set2
chmod 777 dev_set1
head -5711  dev_set2 > equal_num_line
sudo chmod 777 equal_num_line
cat dev_set1 >> equal_num_line 
mv equal_num_line dev.tsv
rm -rf dev_set2
rm -rf dev_set1
sed -i '1 i\index	caption	object	entailment' dev.tsv

sleep 10
echo " DONE "
