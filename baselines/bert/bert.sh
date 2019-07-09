export DATA_DIR=../../../data/bert_classify_thereis_5caps/
export OUTPUT_DIR=../../../logs/bert_finetune/

python3 ../../../bert_baseline/pytorch-pretrained-BERT/examples/run_classifier.py \
  --task_name thereis \
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
