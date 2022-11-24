#!/bin/bash

# IMPL=cached
IMPL=mmap
python /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/tools/preprocess_data_cn.py \
       --input /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/dataset/ \
       --vocab /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/ \
       --dataset_impl ${IMPL} \
       --output_prefix test_samples_${IMPL} \
       --output_path /home/fengty/nlp_workspaces/pycharm_projects/Yuan-1.0/src/megatron/data/test/data_asc \
       --workers 1 \
       --log_interval 2
