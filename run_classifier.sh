#!/usr/bin/env bash
export BERT_BASE_DIR=/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export TASK_NAME=sstlabel2
export CUDA_VISIBLE_DEVICES=3
export TPU_NAME=wuwei

python3 run_classifier.py \
--task_name=${TASK_NAME} \
--do_predict=true \
--data_dir=/data/nfsdata/nlp/datasets/glue_data/SST-2 \
--vocab_file=${BERT_BASE_DIR}/vocab.txt \
--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
--max_seq_length=512 \
--train_batch_size=6 \
--save_checkpoints_steps=2000 \
--iterations_per_loop=2000 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=${TASK_NAME}-output/ \
--use_tpu=false \
--tpu_name=${TPU_NAME} \
--number_labels=1