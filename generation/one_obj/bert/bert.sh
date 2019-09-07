#export DATA_DIR=../../../data/bert_classify_thereis_5caps_seed0/
export DATA_DIR=../../../../../../data/generation_seed0_harder_with_train/
export OUTPUT_DIR=../../../../../../logs/generation/one_obj/bert_logs_harder

python3 run_classifier_multans.py \
  --task_name thereisgen \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $OUTPUT_DIR

