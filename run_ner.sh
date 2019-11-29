#!/usr/bin/env bash
export TASK_NAME=msra
export BERT_BASE_DIR=gs://shannon_storage/bert/chinese_L-12_H-768_A-12/
export DATA_DIR=gs://shannon_storage/sequence_labeling/msra/
export OUTPUT_DIR=gs://shannon_storage/output/msra-bert-crf/
export TPU_NAME=instance-1

python3 run_ner.py \
--task_name=${TASK_NAME} \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=${DATA_DIR} \
--vocab_file=${BERT_BASE_DIR}/vocab.txt \
--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
--output_dir=${OUTPUT_DIR} \
--max_seq_length=512 \
--train_batch_size=512 \
--save_checkpoints_steps=2000 \
--iterations_per_loop=2000 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--use_tpu=true \
--use_crf=true \
--tpu_name=${TPU_NAME}
