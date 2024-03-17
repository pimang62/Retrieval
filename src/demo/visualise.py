import sys
sys.path.append("../../")

# from embeddings.embedding import EmbeddingFactory
from embeddings.huggingface import TransformersEmbedding
# from langchain.vectorstores.faiss import FAISS
import faiss

#### json이나 yaml 파일로 config 읽어서 여기로 넣어주기 
### 옵션별로 결과값 실시간으로 보려고 하기 때문에!@
import umap.umap_ as umap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# 시각화 안되는 문제 해결
import matplotlib
matplotlib.use('Agg')
import pickle

import os
from utils.convert import convert_to_numpy

plt.rcParams['font.family'] = 'NanumGothic'

EMBEDDING_MODEL_NAME_OR_PATH = {
    'bge':"BAAI/bge-large-en-v1.5",
    'korsim':"BM-K/KoSimCSE-bert",
    # 'kordpr': "skt/kobert-base-v1",
    }

# ef = EmbeddingFactory()

def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

# TODO: 3/14 에러 해결
"""
OSError: None is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models' If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
"""

def visualise(query,
              top_k,
              embedding_type,
              chunk_size,
              chunk_overlap,
              database):
    
    # path 설정
    src_dir = "/home/pimang62/projects/ir/a276_document_retrieval/src"
    vectorstore_dir = os.path.join(src_dir, "vectorstore")
    umap_models_dir = os.path.join(src_dir, "umap_models")
    global_path_type = f"{database}_{embedding_type}_{chunk_size}_{chunk_overlap}/{database}"
    
    print("Loading document embeddings (should be saved when creating the vectorstore)")
    with open(f"{vectorstore_dir}/{global_path_type}_embedding.pkl", "rb") as f:
        docs_embeddings = pickle.load(f)
    
    db = faiss.read_index(f"{vectorstore_dir}/{global_path_type}_index.index")
    # db = faiss.index_cpu_to_all_gpus(db)  # OOM 일어날 때는 지우기 -> 일어남
        
    print("Loading umap model (should be saved when creating the vectorstore)")
    with open(f"{umap_models_dir}/{global_path_type}_embedding.umap", "rb") as f:
        umap_embeddings = pickle.load(f)
    with open(f"{umap_models_dir}/{global_path_type}_embedding.umap_model", "rb") as f:
        umap_transform = pickle.load(f)

    te = TransformersEmbedding(model_name=EMBEDDING_MODEL_NAME_OR_PATH[embedding_type])
    query_vector = te.embed_query(query)
    projected_query_embedding = project_embeddings(query_vector, umap_transform)

    scores, docs_index = db.search(convert_to_numpy(query_vector), top_k)

    retrieved_embeddings = [docs_embeddings[i] for i in docs_index.reshape(-1).tolist()]
    projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

    # Plot the projected query and retrieved documents in the embedding space
    fig = plt.figure()
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=10, color='gray')
    plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')
    plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker='X', color='r')

    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'{query}')
    plt.axis('off')
    plt.savefig('../demo/test.png')
    plt.show()
    return fig

if __name__ == "__main__":
    visualise(query="의료 폐기물은 어떻게 처리하나요?",
              top_k=3,
              embedding_type="korsim",
              chunk_size=300,
              chunk_overlap=0,
              database="pdf")