from pymilvus import (
    db,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import os, sys
sys.path.append(os.pardir)
from app.embeddings.openai import OpenAIEmbedding

def create_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, description="ids", is_primary=True, auto_id=False),
        FieldSchema(name="big_category", dtype=DataType.VARCHAR, description="big categories", max_length=10000),
        FieldSchema(name="small_category", dtype=DataType.VARCHAR, description="small categories", max_length=10000),
        FieldSchema(name="question", dtype=DataType.VARCHAR, description="questions", max_length=10000),
        FieldSchema(name="question_emb", dtype=DataType.FLOAT_VECTOR, description="question embeddings", dim=dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, description="contents", max_length=10000),
        # FieldSchema(name="content_emb", dtype=DataType.FLOAT_VECTOR, description="content embeddings", dim=dim),
        # FieldSchema(name="qc_emb", dtype=DataType.FLOAT_VECTOR, description="qc embeddings", dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description=collection_name)  # collection_name
    collection = Collection(name=collection_name, schema=schema)
    return collection

def create_entities(file_path):
    with open(file_path, 'r') as f:
        import json
        data = json.load(f)
    
    documents = data["documents"]  # List[dict()]
    entities = [
        [],  # id
        [],  # big_category
        [],  # small_category
        [],  # question
        [],  # question_emb
        [],  # content
        # [],  # content_emb
        # [],  # qc_emb
    ]  # *len(documents[0])

    for i in range(len(documents)):
        entities[0].append(i)
        entities[1].append(documents[i]["big_category"])
        entities[2].append(documents[i]["small_category"])
        entities[3].append(documents[i]["question"])
        entities[4].append(documents[i]["question_emb"])
        entities[5].append(documents[i]["content"])  
    
    return entities

def query2emb(query):
    embedding_model = OpenAIEmbedding()
    query_emb = embedding_model.embed_query(query)
    # print(f"Query embedding dims length : {len(query_emb)}")  # for debugging
    return query_emb

def search_collection(query_emb):
    global collection
    collection.load()
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }
    result = collection.search(data=[query_emb], anns_field="question_emb",  # query_emb must be 2D
                               param=search_params, limit=3,
                               output_fields=["question", "content"])
    return result

def embed2docs(result):
    documents = []
    for hits in result:
        for hit in hits:
            q, c = hit.entity.get('question'), hit.entity.get('content')
            if q is None:  # exist None
                documents.append(c)
            else:
                documents.append(f'{q}\n{c}')
    return documents

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='/home/pimang62/projects/qachat/Retrieval/data/openai_emb.json')
    parser.add_argument('--collection_name', type=str, default='baemin')

    args = parser.parse_args()
    file_path = args.file_path
    collection_name = args.collection_name
    
    collection = None

    """database connection 하는 부분"""
    connections.connect(alias='default', db_name='mzgpt', host='0.0.0.0', port='19530')
    # print(db.list_database())  # ['default', 'mzgpt']

    """data entities 만드는 부분"""
    entities = create_entities(file_path=file_path)

    """collection 생성하는 부분"""
    collection = create_collection(collection_name=collection_name, dim=1536)
    
    collection.insert(entities)
    
    collection.flush()

    index_params = {
        'index_type':'IVF_FLAT',
        'metric_type':'COSINE',
        'params':{'nlist':128}
    }
    
    collection.create_index(field_name='question_emb', index_params=index_params)
    print("Create indexes Done\n")

    # """query retrieve 하는 부분"""
    # query = "알바생이 계약서를 안쓰면 어떻게 되나요?"
    # query_emb = query2emb(query)

    # result = search_collection(query_emb)

    # documents = embed2docs(result)
    
    # for i, docs in enumerate(documents, 1):
    #     print(f"문서 #{i}\n{docs}", end='\n\n')

