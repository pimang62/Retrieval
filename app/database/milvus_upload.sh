#!/bin/bash

nohup python milvus_upload.py --file_path /home/pimang62/projects/qachat/Retrieval/data/openai_emb.json --collection_name baemin > logs/baemin.log

nohup python milvus_upload.py --file_path /home/pimang62/projects/qachat/Retrieval/data/openai_expert_emb.json --collection_name baemin_expert > logs/baemin_expert.log