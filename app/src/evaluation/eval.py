"""
[dataframe 만들기]
q1 | a1 | [a4 a1 a6] | 2 | 1/2 | 
q2 | a2 | 

"""
import torch
from torch import tensor
import os
import faiss
from faiss import read_index
import numpy as np
import pandas as pd

import os, sys
sys.path.append('../../')

from app.embeddings.huggingface import TransformersEmbedding
from app.utils.convert import convert_to_numpy 

embedding_model = None
def get_score_and_index(database,
                        embedding_type,
                        chunk_size,
                        chunk_overlap,
                        query, 
                        top_k):
    global embedding_model
    """Return faiss relavant score and index"""
    database_name = database
    if embedding_type == 'bge':
        embedding_model_name_or_path = "BAAI/bge-large-en-v1.5"
    elif embedding_type == 'korsim':
        embedding_model_name_or_path = "BM-K/KoSimCSE-bert"
    elif embedding_type == 'kordpr':
        embedding_model_name_or_path = "skt/kobert-base-v1"
    
    if embedding_model == None:
        embedding_model = TransformersEmbedding(model_name=embedding_model_name_or_path)
    
    query_emb = embedding_model.embed_query(query)

    vectorstore_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "vectorstore")
    emb_dir_path = f"{database_name}_{embedding_type}_{chunk_size}_{chunk_overlap}"
    db = faiss.read_index(f"{vectorstore_path}/{emb_dir_path}/{database_name}_index.index")
    score, index = db.search(convert_to_numpy(query_emb), top_k)

    return score, index

def calculate_rank(y_pred, y_true):
    """Calculate rank and inverse rank
       Return rank, 1/rank 
    """
    rank_index = torch.nonzero(y_pred==y_true, as_tuple=False)  # [[1]] index

    if not len(rank_index):   # 같은 인덱스가 하나도 없다면
        return 0, 0
    else:
        rank = rank_index.squeeze().item() + 1
        return rank, 1/rank

# def get_retrieve_documents(database, ):
#     vectorstore = VectorStore(database_name=database, 
#                               embedding_model_name_or_path=embedding_type, 
#                               chunk_size=chunk_size, 
#                               chunk_overlap=chunk_overlap)
    
#     texts = vectorstore.get_chunked_texts()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="baemin")
    parser.add_argument("--embedding_type", type=str, default="korsim")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=0)

    args = parser.parse_args()
    database = args.database
    embedding_type = args.embedding_type
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    top_k = args.top_k

    with open('/home/pimang62/projects/ir/a276_document_retrieval/data/txt.txt', 'r') as f:
        questions = f.read()
    
    retrieve_list = []
    rank_list = []
    inverse_rank_list = []
    for idx, question in enumerate(questions.split('\n\n')):
        _, indices = get_score_and_index(database=database,
                            embedding_type=embedding_type,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            query=question, 
                            top_k=top_k)
        
        retrieve_list.append(indices)
        
        y_pred = tensor(indices).squeeze()  # tensor([77, 35, 127])
        y_true = tensor(idx)
        
        # rank와 1/rank 계산하기
        rank, inverse_rank = calculate_rank(y_pred, y_true)

        rank_list.append(rank) 
        inverse_rank_list.append(inverse_rank)
    
    df = pd.DataFrame({'question_index': range(len(retrieve_list)),
                       'retrieve_indices': retrieve_list,
                       'rank': rank_list,
                       '1/rank': inverse_rank_list})
    print(df.head(10))
    # print(len(df))  # 137
    # print(len(df[df['1/rank'] == 0]))
    # print(df.info())
    # print(df['rank'].unique())  # 0 1 2 3
    # print(df['1/rank'].unique())  # 1 0.5 0.33

    print(f"MRR score is {df['1/rank'].mean()}")

