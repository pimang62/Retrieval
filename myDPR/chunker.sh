#!/bin/bash

# python chunker.py --data_path "dataset/baemin.json" \
#                   --passages_dir "baemin_passages_" \
#                   --title_index_map_path "baemin_title_passage_map_.p" \
#                   --passages_type "" \
#                   --chunk_size 100 \
#                   --chunk_overlap 50

python chunker.py --data_path "dataset/baemin.json" \
                  --passages_dir "baemin_passages_answer" \
                  --title_index_map_path "baemin_title_passage_map_answer.p" \
                  --passages_type "answer" \