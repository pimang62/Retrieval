#!/bin/bash
databases=("txt" "pdf" "docx")  #  했음
embedding_types=("bge" "korsim" "openai") 
chunk_size=(100 300 1000)

PROJECT_ROOT="/home/pimang62/projects/ir/Retrieval"

for database in "${databases[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        for chunk in "${chunk_size[@]}"; do
            python3 -m app.src.demo.vectorstore --database $database \
                                    --fname "$PROJECT_ROOT/app/data/${database}.${database}" \
                                    --embedding_type $embedding \
                                    --chunk_size $chunk \
                                    --chunk_overlap 0
        done
    done
done

