#! /bin/bash

# This script is single-node multi-GPU

#GPUS_PER_NODE=2
#NNODES=1
#NODE_RANK=0
#MASTER_ADDR=localhost
#MASTER_PORT=6000

#WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/checkpoint
DATA_PATH=/home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/megatron/data/test/data_asc/001.txt_document_context

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

GPT_ARGS="--num-layers 24 \
      --hidden-size 1024 \
      --num-attention-heads 16 \
      --seq-length 1024 \
      --max-position-embeddings 1024 \
      --micro-batch-size 8 \
      --global-batch-size 64 \
      --lr 0.00015 \
      --train-iters 500000 \
      --lr-decay-iters 320000 \
      --lr-decay-style cosine \
      --tokenizer-type EncDecTokenizer \
      --vocab-file /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/vocab.txt \
      --merge-file /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/gpt2-merges.txt \
      --lr-warmup-fraction .01 \
      --fp16"

OUTPUT_ARGS="--log-interval 100 \
      --save-interval 2000 \
      --eval-interval 100 \
      --eval-iters 10 \
      --checkpoint-activations \
"

#python -m torch.distributed.launch $DISTRIBUTED_ARGS \
python -m torch.distributed.run --nproc_per_node 2 \
       /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/pretrain_gpt.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --split 95,3,2 \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0

#---------------------------------------------------------------
#       --data-impl              数据集索引方式
#       --split                  用逗号分隔的比例列表，用于训练、验证和测试拆分。例如，分割' 90,5,5 '将使用90%%的数据进行训练，5%用于验证，5%用于测试。
#       --min-lr                 学习率的最小值
#       --weight-decay           Weight decay coefficient for L2 regularization
#       --clip-grad              Gradient clipping based on global L2 norm.
#---------------------------------------------------------------
