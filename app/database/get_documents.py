from typing import List, Dict, Any, Literal

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import os
import sys
sys.path.append(os.pardir)
from app.embeddings.openai import OpenAIEmbedding

import numpy as np

def search_collection(
        collection, 
        query_vec, 
        field_name,
        metric_type, 
        top_k
) -> object:

    collection.load()
    search_params = {
        "metric_type": f"{metric_type}",
        "params": {"nprobe": 10},
    }
    result = collection.search([query_vec], anns_field=field_name,  # query_vec must be 2D
                                param=search_params, limit=top_k,
                                output_fields=["question", "content"])
    return result

def embed2docs(result: object) -> List[tuple[int, str]]:
    documents = []
    for hits in result:
        for hit in hits:
            q, c = hit.entity.get('qustion'), hit.entity.get('content')
            score = hit.distance
            if q is None:  # exist None
                documents.append((score, c))
            else:
                documents.append((score, f'{q}\n{c}'))
    return documents

def retrieve_docs_with_score(
    db_name: str,  # 'default'
    collection_name: str,
    field_name: str,
    method: Literal["l2_distance", "cosine_similarity"],
    query_vec: List[float] = None,
    top_k: int = 3,
) -> List[tuple[int, str]]:
    """
    example:
        retrieve_docs_with_score(
            db_name='mzgpt',  # 'default'
            collection_name='baemin',
            field_name='qc_emb'
            method='cosine_similarity',
            query_vec=query_vec,
            top_k=3,
        )
    """

    connections.connect(alias='default', db_name=db_name, host='0.0.0.0', port='19530')
    collection = Collection(collection_name)
    
    """Retrieve documents by methods like cosine similarity"""
    if method == "l2_distance":
        metric_type = "L2"
    elif method == "cosine_similarity":
        metric_type = "COSINE"

    result = search_collection(collection, query_vec, field_name, metric_type, top_k)
    documents = embed2docs(result)
    
    return documents

if __name__ == '__main__':
    
    query = "알바생이 계약서를 안쓰면 어떻게 되나요?"  # input()

    embedding_model = OpenAIEmbedding()
    query_vec = embedding_model.embed_query(query)

    # doc1 = retrieve_docs_with_score(
    #     alias='default',
    #     collection_name='baemin',
    #     method='l2_distance',
    #     query_vec=query_vec,
    #     top_k=3,
    # )

    doc2 = retrieve_docs_with_score(
        db_name='mzgpt',
        collection_name='baemin_expert',
        field_name='qc_emb',
        method='cosine_similarity',
        query_vec=query_vec,
        top_k=3,
    )
    
    doc3 = retrieve_docs_with_score(
        db_name='mzgpt',
        field_name='question_emb',
        collection_name='baemin_expert_qonly',
        method='cosine_similarity',
        query_vec=query_vec,
        top_k=3,
    )


    # for i, docs in enumerate(doc1, 1):
    #     score, document = docs
    #     print(f"문서 #{i}, distance: {score*100}\n{document}", end='\n\n')
    
    # print('---------------------------------------------------\n')

    for i, docs in enumerate(doc2, 1):
        score, document = docs
        print(f"문서 #{i}, similarity: {score*100:.2f}\n{document}", end='\n\n')
    
    print('---------------------------------------------------\n')

    for i, docs in enumerate(doc3, 1):
        score, document = docs
        print(f"문서 #{i}, similarity: {score*100:.2f}\n{document}", end='\n\n')