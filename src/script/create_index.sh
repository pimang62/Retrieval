#!/bin/bash
databases=("txt" "pdf" "docx")  #  했음
embedding_types=("bge" "korsim")  # "openai"
chunk_size=(100 300 1000)

for database in "${databases[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        for chunk in "${chunk_size[@]}"; do
            python ../demo/vectorstore.py --database $database \
                                    --fname "/home/pimang62/projects/ir/a276_document_retrieval/data/${database}.${database}" \
                                    --embedding_type $embedding \
                                    --chunk_size $chunk \
                                    --chunk_overlap 0
        done
    done
done

python vectorstore.py --database "txt" \
                      --fname "/home/pimang62/projects/ir/a276_document_retrieval/data/openai_emb.json" \  # 다시 update해야
                      --embedding_type "openai" \
                      --chunk_size 100 \  # 다시 update해야
                      --chunk_overlap 0 

python vectorstore.py --database "txt" \
                      --fname "/home/pimang62/projects/ir/a276_document_retrieval/data/openai_expert_emb.json" \ 
                      --embedding_type "openai" \
                      --chunk_size 100 \  # "
                      --chunk_overlap 0 
