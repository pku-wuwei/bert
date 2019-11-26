#!/usr/bin/env bash
export BERT_BASE_DIR=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12
export TASK_NAME=20news
export CUDA_VISIBLE_DEVICES=3
export TPU_NAME=wuwei

python3 run_interaction.py \
--task_name=${TASK_NAME} \
--do_train=true \
--do_eval=true \
--do_predict=true \
--is_interactive=true \
--data_dir=/data/nfsdata/nlp/datasets/classification/hedwig/20news \
--vocab_file=${BERT_BASE_DIR}/vocab.txt \
--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
--max_seq_length=512 \
--train_batch_size=6 \
--save_checkpoints_steps=2000 \
--iterations_per_loop=2000 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=${TASK_NAME}-bert-interactive/ \
--use_tpu=false \
--tpu_name=${TPU_NAME}
