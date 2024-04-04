# HuggingFace model for embedding
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import torch

from app.embeddings.base import Embeddings

EMBEDDING_MODEL_NAME_OR_PATH = {
    'bge':"BAAI/bge-large-en-v1.5",
    'korsim':"BM-K/KoSimCSE-bert",
    # 'kordpr': "skt/kobert-base-v1",
    }

DEFAULT_MODEL_NAME = "BM-K/KoSimCSE-bert"

class TransformersEmbedding(BaseModel, Embeddings):
    model: str = None
    tokenizer: Any = None
    model_name: str = None
    #cache_folder: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, **kwargs: Any):
        """Initialize transformers model"""
        super().__init__(**kwargs)
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Could not import transformers. "
                "Please install it with `pip install transformers`."
            ) from exc

        print("Loading transformers model", self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_embedding(self, embedding_type):
        embedding_model_name_or_path = EMBEDDING_MODEL_NAME_OR_PATH[embedding_type]
        model_kwargs = {'device': 'cuda:2'}
        # self.encode_kwargs = {'convert_to_tensor': True}
        embeddings = TransformersEmbedding(
            model_name=embedding_model_name_or_path,
            model_kwargs=model_kwargs,
            # encode_kwargs=encode_kwargs
        )
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute document embeddings"""
        texts = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        embeddings, _ = self.model(**texts, return_dict=False)
        return [vector[0].tolist() for vector in embeddings]  # 2D list

    def embed_query(self, text: str) -> List[List[float]]:
        """Compute query embeddings"""
        text = self.tokenizer(
            [text], 
            truncation=True,
            padding=True,
            return_tensors="pt",
            )
        embeddings, _ = self.model(**text, return_dict=False)
        return [vector[0].tolist() for vector in embeddings]


# if __name__ == "__main__":  
#     te = TransformersEmbedding(model_name="BAAI/bge-large-en-v1.5")
#     print(te.embed_documents(["안녕하세요", "반갑습니다"])[0][:10])

#     import sys
#     sys.path.append("/mnt/data1/pimang62/ir/a276_document_retrieval")
#     from embedding import EmbeddingFactory
#     ef = EmbeddingFactory()
#     emd = ef.load_embedding('kordpr')
#     query_dims, query_vector = emd.embed_query("대체 발급된 안심 번호는 사용")
#     import pdb; pdb.set_trace()