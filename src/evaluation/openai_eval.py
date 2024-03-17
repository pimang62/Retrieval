"""
openai_eval.sh 실행
"""

import torch
from torch import tensor
import os
import faiss
from faiss import write_index, read_index

import sys
sys.path.append('../../')
from utils import convert_to_numpy

import json
from tqdm import tqdm
import pandas as pd

root_dir = os.pardir
 
queries = []  # global

openai_dir_path = None  # 갱신
openai_index_path = None  
vectorstore_path = os.path.join(root_dir, "vectorstore")

def build_openai_index():
    openai_index = faiss.IndexFlatL2(len(embeddings[0]))  # 2D[0] -> 1D
    
    faiss.normalize_L2(convert_to_numpy(embeddings))
    openai_index.add(convert_to_numpy(embeddings))  # nohead

    os.makedirs(os.path.join(vectorstore_path, openai_dir_path), exist_ok=True)
    faiss.write_index(openai_index, openai_index_path)

    print("Successful writing faiss index\n")
    return openai_index


def get_openai_score_and_index(query, top_k):
    open_db = faiss.read_index(openai_index_path)
    score, indices = open_db.search(convert_to_numpy(query), top_k)
    return score, indices


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="baemin_nohead")
    parser.add_argument("--top_k", type=int, default=3)
    
    args = parser.parse_args()
    database = args.database
    top_k = args.top_k
    
    openai_dir_path = f"{database}_openai_{100}_{0}"  # chunk 100, overlap 0 fix
    openai_index_path = f"{vectorstore_path}/{openai_dir_path}/{database}_index.index"
    
    with open('../../data/openai_emb.json', 'r') as f:
        content = json.load(f)
        documents = content['documents']  # 137 dict
    
    queries, embeddings = [], []  # nohead / head
    for info_dict in documents:
        queries.append(info_dict["question_emb"])
        if "nohead" in database:  # str
            embeddings.append(info_dict["content_emb"])  # 137
        else:  # 전체 context
            embeddings.append(info_dict["qc_emb"])  # 137
    
    build_openai_index()

    rank_list = []
    inverse_rank_list = []
    retrieve_list = []
    for idx, query in enumerate(queries):
        
        _, indices = get_openai_score_and_index(convert_to_numpy([query]), top_k)
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

    print(f"MRR score is {df['1/rank'].mean()}")