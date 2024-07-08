from abc import ABC, abstractmethod
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from app.model.kobert_biencoder import KobertBiEncoder
from kobert_tokenizer import KoBERTTokenizer
import torch

from typing import List

import os
# 현재 파일의 위치를 기준으로 상위 디렉토리의 절대 경로를 계산
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

class Embedding(ABC):
    @abstractmethod
    def load_embedding(self):
        pass

class BGEEmbedding(Embedding):
    
    def load_embedding(self):
        self.embedding_model_name = "BAAI/bge-large-en-v1.5"
        self.model_kwargs = {'device': 'cuda:2'}
        self.encode_kwargs = {'convert_to_tensor': True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=self.model_kwargs,
            # encode_kwargs=encode_kwargs
        )
        return self.embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.embeddings.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Dimension of the embeddings
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.embeddings.encode(
            self.query_instruction + text, **self.encode_kwargs
        )
        vector = embedding.tolist()
        return vector


class KorSimEmbedding(Embedding):
    def load_embedding(self):
        embedding_model_name = "BM-K/KoSimCSE-bert"
        model_kwargs = {'device': 'cuda:2'}
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
        )
        return embeddings

class KorDPREmbedding(Embedding, Embeddings):
    def __init__(self) -> None:
        super().__init__()
        

    def load_embedding(self):
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        
        self.model = KobertBiEncoder()
        self.model.load(f"{root_dir}/model/kobert_biencoder.pt")
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        
        return self

    def embed_query(self, query: str) -> List[float]:
        tok = self.tokenizer.batch_encode_plus([query])
        tok = {k: torch.tensor(v).to(self.device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.model(torch.tensor(tok["input_ids"]), torch.tensor(tok["attention_mask"]), "query")

        return out.cpu().flatten().tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        tok = self.tokenizer.batch_encode_plus(documents, padding="longest") # padding="max_length" max_length=512
        tok = {k: torch.tensor(v).to(self.device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.model(torch.tensor(tok["input_ids"]), torch.tensor(tok["attention_mask"]), "query")

        return out.cpu().flatten().tolist()


class EmbeddingFactory:
    def load_embedding(self, embedding_name):
        if embedding_name == "bge":
            return BGEEmbedding().load_embedding()
        elif embedding_name == "korsim":
            return KorSimEmbedding().load_embedding()
        elif embedding_name == "kordpr":
            return KorDPREmbedding().load_embedding()
        else:
            raise NotImplementedError(f"Embedding {embedding_name} not implemented")
        
if __name__ == "__main__":
    ef = EmbeddingFactory()
    emd = ef.load_embedding('kordpr')
    query_vector = emd.embed_query("대체 발급된 안심 번호는 사용")
    import pdb; pdb.set_trace()