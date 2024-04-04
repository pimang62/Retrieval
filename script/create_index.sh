#!/bin/bash
databases=("txt" "pdf" "docx")  #  했음
embedding_types=("bge" "korsim" "openai") 
chunk_size=(100 300 1000)

for database in "${databases[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        for chunk in "${chunk_size[@]}"; do
            python ../app/src/demo/vectorstore.py --database $database \
                                    --fname "../app/data/${database}.${database}" \
                                    --embedding_type $embedding \
                                    --chunk_size $chunk \
                                    --chunk_overlap 0
        done
    done
done

