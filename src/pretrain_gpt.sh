#! /bin/bash

# this script is used to run my own gpt2(345M) model

CHECKPOINT_PATH=/home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/checkpoint
DATA_PATH=/home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/megatron/data/test/data_asc/001.txt_document_context

GPT_ARGS="--num-layers 24 \
          --hidden-size 1024 \
          --num-attention-heads 16 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 4 \
          --global-batch-size 8 \
          --lr 0.00015 \
          --train-iters 500000 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --tokenizer-type EncDecTokenizer \
          --vocab-file /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/vocab.txt \
          --merge-file /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/gpt2-merges.txt \
          --lr-warmup-fraction .01 \
          --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/pretrain_gpt.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \

# --------------------------参数详解------------------------------

# --save                      要保存检查点的输出目录
# --load                      模型检查点的目录
# --data-path                 训练数据集的路径

# --------------------------GPT_ARGS----------------------------

#  --num-layers               transformer层数
#  --hidden-size              tarnsfromer hidden size
#  --num-attention-heads      多头注意力机制头的数量 Number of transformer attention heads.
#  --seq-length               要处理的最大序列长度 Maximum sequence length to process.
#  --max-position-embeddings  Maximum number of position embeddings to use.
#  --micro-batch-size         Batch size per model instance (local batch size). Global batch size is local batch size times data parallel size times number of micro batches.
#  --global-batch-size        Training batch size. If set, it should be a multiple of micro-badtch-size times data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size as the global batch size. This choice will result in 1 for number of micro-batches.
#  --lr                       Initial learning rate. Depending on decay style and initial warmup, the learing rate at each iteration would be different.
#  --train-iters              Total number of iterations to train over all training runs. Note that either train-iters or train-samples should be provided.'
#  --lr-decay-iters           number of iterations to decay learning rate over, If None defaults to `--train-iters`
#  --lr-decay-style           Learning rate decay function. choices:['constant', 'linear', 'cosine']
#  --vocab-file               词汇文件路径
#  --merge-file               BPE合并文件的路径
#  --lr-warmup-fraction       fraction of lr-warmup-(iters/samples) to use for warmup (as a float)
#  --fp16                     以fp16模式运行模型

# -------------------------OUTPUT_ARGS---------------------------

# --log-interval              Report loss and timing interval
# --save-interval             Number of iterations between checkpoint saves
# --eval-interval             Interval between running evaluation on validation set
# --eval-iters                Number of iterations to run for evaluation validation/test for
# --checkpoint-activations    Checkpoint activation to allow for training with larger models, sequences, and batch sizes