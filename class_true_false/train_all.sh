cd basic_model
#python3 train_basic_model.py
#echo "trained base cross ent"

#cd ../attention_embedding
#python3 train.py
#echo "trained attention_embedding"

#cd ../reuse_hidden_state
#python3 train.py
#echo "trained reuse_state"

#cd ../inner_attention
#python3 train.py
#echo "trained inner attention"

#cd ../embedding
#python3 train.py
#echo "trained embedding"
#
#cd ../attention
#python3 train_att_model.py
#echo "trained attention"

#cd ../no_cap
#python3 train.py
#echo "trained no_cap"

#cd ../no_obj
#python3 train.py
#echo "trained no_obj"

cd ../preload_emb
python3 train.py
echo "trained preload glove"
