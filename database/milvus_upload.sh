#!/bin/bash

# nohup python baemin_upload.py --file_path /home/pimang62/projects/qachat/a245_chatgpt_knowledge_chatbot/api/json/openai_emb.json --collection_name baemin > logs/baemin.log 2>&1 &

# nohup python baemin_upload.py --file_path /home/pimang62/projects/qachat/a245_chatgpt_knowledge_chatbot/api/json/openai_expert_emb.json --collection_name baemin_expert > logs/baemin_expert.log 2>&1 &

nohup python baemin_upload.py --file_path /home/pimang62/projects/qachat/a245_chatgpt_knowledge_chatbot/api/json/openai_expert_emb.json --collection_name baemin_expert_qonly > logs/baemin_expert_qonly.log 2>&1 &