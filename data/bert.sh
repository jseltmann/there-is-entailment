export DATA_DIR=/home/users/srauniyar/data/entailment_data_analysis/obj_in_caption/score
#export DATA_DIR=/data/entailment_data_analysis/obj_in_caption/cap_not_seen
#export DATA_DIR=/data/entailment_data_analysis/bert/my_tests
export OUTPUT_DIR=/data/entailment_data_analysis/logs

#python3 /home/users/kuan/run_classifier.py
python3 /home/users/jseltmann/there-is-entailment/bert_baseline/pytorch-pretrained-BERT/examples/run_classifier_preds.py \
  --task_name thereis \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model /home/users/jseltmann/there-is-entailment/logs/bert_classify_1epoch_seed0/ \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --output_dir $OUTPUT_DIR
