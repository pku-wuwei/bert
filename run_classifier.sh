#!/usr/bin/env bash
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=gs://shannon_albert/hedwig
export TASK_NAME=Reuters-Wiki
export STORAGE_BUCKET=gs://shannon_albert
export TPU_NAME=xiaoyli

python3 run_classifier.py \
--task_name=${TASK_NAME} \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=${GLUE_DIR}/Reuters \
--vocab_file=${BERT_BASE_DIR}/vocab.txt \
--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
--max_seq_length=512 \
--train_batch_size=128 \
--save_checkpoints_steps=20000 \
--iterations_per_loop=20000 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=${STORAGE_BUCKET}/${TASK_NAME}-output/ \
--use_tpu=True \
--tpu_name=${TPU_NAME}