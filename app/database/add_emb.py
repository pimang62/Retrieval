import json
from pathlib import Path

import sys
sys.path.append("/home/pimang62/projects/ir/a276_document_retrieval/")
from app.embeddings.openai import OpenAIEmbedding

load_path = '/home/pimang62/projects/qachat/Retrieval/demo/data/merge_requested_form.json'
save_path = '/home/pimang62/projects/qachat/Retrieval/demo/data/openai_expert_emb.json'
# save_path = '/home/pimang62/projects/ir/a276_document_retrieval/demo/data/openai_expert_emb_1.json'

with open(load_path, 'r') as f:
    print("Starting make embeddings ...")

    data = json.load(f)
    embedding = OpenAIEmbedding()

    docs = data["documents"]
    for i in range(len(docs)):
        # string
        question = docs[i]["question"]
        content = docs[i]["content"]
        # list
        question_emb = embedding.embed_query(question)
        content_emb = embedding.embed_query(content)
        qc_emb = embedding.embed_query('\n'.join([question, content]))

        docs[i]["question_emb"] = question_emb
        docs[i]["content_emb"] = content_emb
        docs[i]["qc_emb"] = qc_emb
    
    print("Making embedding is done.")

with open(save_path, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

    print("Saving file is done.")

