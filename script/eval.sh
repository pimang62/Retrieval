#!/bin/bash
 
databases=("txt")  # baemin_nohead" "baemin
embedding_types=("bge" "korsim")

for db in "${databases[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        python ../evaluation/eval.py --database $db \
                       --embedding_type $embedding \
                       --chunk_size 100 \
                       --chunk_overlap 0 \
                       --top_k 3
    done
done
