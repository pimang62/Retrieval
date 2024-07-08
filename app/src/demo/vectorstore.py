"""
VectorStore

1. Index Builder
2. Retriever
"""
import os

from app.chunker.data_chunk import ChunkerFactory
from app.embeddings.huggingface import TransformersEmbedding
from app.embeddings.openai import OpenAIEmbedding

import faiss
from faiss import write_index, read_index
import numpy as np
from app.utils.convert import convert_to_numpy  # List[List[float]] -> np.ndarray

import os
import pickle
import umap.umap_ as umap
from tqdm import tqdm

root_dir = "/home/pimang62/projects/ir/Retrieval/app/src"

def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform(convert_to_numpy([embedding]))
    return umap_embeddings

EMBEDDING_MODEL_NAME_OR_PATH = {
    'bge':"BAAI/bge-large-en-v1.5",
    'korsim':"BM-K/KoSimCSE-bert",
    # 'kordpr': "skt/kobert-base-v1",
    }

class VectorStore:
    def __init__(self, database_name, embedding_model_name_or_path, chunk_size, chunk_overlap):  # <- fname X
        self.database_name = database_name
        self.embedding_model_name = embedding_model_name_or_path  # key
        self.embedding_model_path = None  # value

        self.chunker = ChunkerFactory(database_name).create_chunker()
        if self.embedding_model_name in EMBEDDING_MODEL_NAME_OR_PATH:  # huggingface
            self.embedding_model_path = EMBEDDING_MODEL_NAME_OR_PATH[self.embedding_model_name]  # 갱신
            self.embedding_model = TransformersEmbedding(model_name=self.embedding_model_path)
        else:  # openai
            self.embedding_model = OpenAIEmbedding()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        #### faiss_index 저장 디렉토리 만들기
        self.vectorstore_path = os.path.join(root_dir, "vectorstore")
        os.makedirs(self.vectorstore_path, exist_ok=True)

        #### umap_models 저장 디렉토리 생성
        self.umap_dir_path = os.path.join(root_dir, "umap_models")
        os.makedirs(self.umap_dir_path, exist_ok=True)  

        #### embedding 타입에 맞는 하위 디렉토리 생성 : {database}_{embedding_type}_{chunk_size}_{chunk_overlap} 형태
        self.emb_dir_path = f"{self.database_name}_{self.embedding_model_name}_{self.chunk_size}_{self.chunk_overlap}"
        os.makedirs(os.path.join(self.vectorstore_path, self.emb_dir_path), exist_ok=True)  # ~vectorstore/baemin_bge_100_0
        os.makedirs(os.path.join(self.umap_dir_path, self.emb_dir_path), exist_ok=True)  # ~umap_models/baemin_bge_100_0

        """chunk texts"""
        self.texts = self.chunker.chunk(chunk_size=self.chunk_size, 
                                   chunk_overlap=self.chunk_overlap)

    def get_chunked_texts(self):
        return self.texts

    def build_index(self, distance_metric):
        """
        Build FAISS index 
        - create index with embedding model

        Returns:
            faiss index object
        """
        self.embeddings = self.embedding_model.embed_documents(self.texts)

        #### faiss index 생성 : IndexFlatL2 (Euclidean) == IndexFlatIP (Cosine)
        self.faiss_index = faiss.IndexFlatL2(convert_to_numpy(self.embeddings).shape[-1])  # dimension
        
        if distance_metric == 'cosine':
            faiss.normalize_L2(convert_to_numpy(self.embeddings))  # cosine
        self.faiss_index.add(convert_to_numpy(self.embeddings))
        
        index_path = f"{self.vectorstore_path}/{self.emb_dir_path}/{self.database_name}_index.index"
        faiss.write_index(self.faiss_index, index_path) 

        print("Successful writing faiss index\n")
        return self.faiss_index
    
    def get_embeddings(self):
        return self.embeddings  # List[List[str]]

    def retrieve(self, query, top_k): 
        """Search KNN with query vector"""
        query_emb = self.embedding_model.embed_query(query)
        
        db = faiss.read_index(f"{self.vectorstore_path}/{self.emb_dir_path}/{self.database_name}_index.index")
        score, index = db.search(convert_to_numpy(query_emb), top_k)
    
        # print("Loading document embeddings (should be saved when creating the vectorstore)")
        # with open(f"{self.vectorstore_path}/{self.emb_dir_path}/{self.database_name}_embedding.pkl", "rb") as f:
        #     docs_embeddings = pickle.load(f)
        
        retrieve_documents = [self.texts[i] for i in index.reshape(-1).tolist()]  # 2d -> 1d  error

        return retrieve_documents
    
    def save_vectorstore(self):
        """
        Save vectorstore in local
        (mongodb로 업로드 추후)
        """
        with open(f"{self.vectorstore_path}/{self.emb_dir_path}/{self.database_name}_embedding.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)
        
        save_umap_embeddings = True
        if save_umap_embeddings:
            
            umap_transform = umap.UMAP().fit(convert_to_numpy(self.embeddings))
            projected_dataset_embeddings = project_embeddings(self.embeddings, umap_transform)
        
            ## 이 작업은 본 스크립트가 끝나고도 실행되도록 뒷단으로 넘겨야함
            print("Projecting umap embeddings for the database.")
            with open(f"{self.umap_dir_path}/{self.emb_dir_path}/{self.database_name}_embedding.umap", "wb") as f:
                pickle.dump(projected_dataset_embeddings, f)
            print(f"Saving trained model.")
            with open(f"{self.umap_dir_path}/{self.emb_dir_path}/{self.database_name}_embedding.umap_model", "wb") as f:
                pickle.dump(umap_transform, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="baemin")
    parser.add_argument("--fname", type=str, default="/home/pimang62/projects/ir/a276_document_retrieval/demo/data/txt.txt")
    parser.add_argument("--embedding_type", type=str, default="bge")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--chunk_overlap", type=int, default=0)

    args = parser.parse_args()
    database = args.database
    fname = args.fname
    embedding_type = args.embedding_type
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    vectorstore = VectorStore(database_name=database, 
                              # fname=fname,
                              embedding_model_name_or_path=embedding_type, 
                              chunk_size=chunk_size, 
                              chunk_overlap=chunk_overlap)
    
    # Index Builder
    faiss_index = vectorstore.build_index(distance_metric='cosine')

    # Save embeddings
    vectorstore.save_vectorstore()

    ## Retriever
    # retrieve_documents = vectorstore.retrieve(query="대체 발급된 안심 번호는 사용할 수 있는 기한이 언제야?",
    #                                           top_k=3)
    
    # print(retrieve_documents)