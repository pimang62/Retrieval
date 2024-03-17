# Abstract class for Embedding models
from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Abstract class for embedding models"""

    @abstractmethod
    def embed_documents(self, text_list: List[str]) -> List[List[float]]:
        """Embed documents"""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query"""
