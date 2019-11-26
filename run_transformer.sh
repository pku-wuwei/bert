#!/usr/bin/env bash
export BERT_BASE_DIR=/data/nfsdata2/home/wuwei/study/bert/transformer_config
export TASK_NAME=20news
export CUDA_VISIBLE_DEVICES=2
export TPU_NAME=wuwei

python3 run_interaction.py \
--task_name=${TASK_NAME} \
--do_train=true \
--do_eval=true \
--do_predict=true \
--is_interactive=false \
--data_dir=/data/nfsdata/nlp/datasets/classification/hedwig/20news \
--vocab_file=${BERT_BASE_DIR}/vocab.txt \
--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
--max_seq_length=512 \
--train_batch_size=6 \
--save_checkpoints_steps=2000 \
--iterations_per_loop=2000 \
--learning_rate=1e-6 \
--num_train_epochs=2000.0 \
--output_dir=${TASK_NAME}-transformer/ \
--use_tpu=false \
--tpu_name=${TPU_NAME}
