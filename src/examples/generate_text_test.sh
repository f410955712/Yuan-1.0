#!/bin/bash

python /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/tools/generate_samples_gpt2.py \
       --tensor-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --load /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/checkpoint/ \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --tokenizer-type EncDecTokenizer \
       --fp16 \
       --micro-batch-size 2 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/vocab.txt \
       --merge-file /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/gpt2-merges.txt \
       --top_p 0.9 \
       --sample-output-file "/home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/input/poetry_extreme_output.txt" \
       --sample-input-file "/home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/input/poetry_extreme_input.txt" \
       --recompute
