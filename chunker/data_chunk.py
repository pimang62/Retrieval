from abc import ABC, abstractmethod
from chunker.base import Chunker

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace, clean_dashes, group_broken_paragraphs 
import re
import os

def preprocess(text: str):
    """
    Remove duplicated white spaces from text

    example.
    '       ' -> ' '
    """
    return re.sub('\s+', ' ', text).strip()

# if you want to split Q & A
def split_contents(doc: str) -> tuple:  
    """Split question and content by newline"""
    question, *content = doc.split('\n')
    return preprocess(question), preprocess(' '.join(content))


class TXTChunker(Chunker):
    """Chunk baemin.txt data"""
    def __init__(self, fname):
        super().__init__(fname)
        self.doc = None

    def load(self):
        loader = TextLoader(self.fname)  # Use self.fname
        self.doc = loader.load()
        self._is_loaded = True  
        print(f"Number of documents: {len(self.doc)}")

    def chunk(self, chunk_size, chunk_overlap):
        """
        Returns:
            texts (List[str]): list of texts
        """
        if not self._is_loaded:  # Check if data is loaded
            self.load() 
         
        # baemin.txt is delimited by "\n\n"
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"],
                                                       chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
        
        texts = [preprocess((docs.page_content)) for docs in text_splitter.split_documents(self.doc)]  # text_splitter.split_documents(self.doc) : [{Document: ...}, {}, ...]
        
        return texts


class PDFChunker(Chunker):
    """Chunk *.pdf data"""
    def __init__(self, fname):
        super().__init__(fname)
        self.doc = None

    def load(self):
        loader = UnstructuredFileLoader(self.fname, mode="single",
                                        # post_processors=[clean_bullets, 
                                        #                  clean_extra_whitespace, 
                                        #                  clean_dashes, 
                                        #                  group_broken_paragraphs]
                                        )
        self.doc = loader.load()
        self._is_loaded = True
        print(f"Number of documents: {len(self.doc)}")

    def chunk(self, chunk_size, chunk_overlap):
        """
        Returns:
            texts (List[str]): list of texts
        """
        if not self._is_loaded:  # Check if data is loaded
            self.load() 
        
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        texts = [preprocess((docs.page_content)) for docs in text_splitter.split_documents(self.doc)]

        return texts


class DocxChunker(Chunker):
    """Chunk *.docx data"""
    def __init__(self, fname):
        super().__init__(fname)
        self.doc = None

    def load(self):
        loader = UnstructuredFileLoader(self.fname, mode="single",
                                        # post_processors=[clean_bullets, 
                                        #                  clean_extra_whitespace, 
                                        #                  clean_dashes, 
                                        #                  group_broken_paragraphs]
                                        )
        self.doc = loader.load()
        self._is_loaded = True
        print(f"Number of documents: {len(self.doc)}")

    def chunk(self, chunk_size, chunk_overlap):
        """
        Returns:
            texts (List[str]): list of texts
        """
        if not self._is_loaded:  # Check if data is loaded
            self.load() 
        
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        texts = [preprocess((docs.page_content)) for docs in text_splitter.split_documents(self.doc)]

        return texts

## chunk factory
class ChunkerFactory:
    """Factory for chunking"""
    def __init__(self, database_name):  # <- fname X
        self.database_name = database_name
        # self.fname = fname
        self.fname = os.path.join("/home/pimang62/projects/ir/a276_document_retrieval/data", f"{self.database_name}.{self.database_name}")

    def create_chunker(self):
        if self.database_name in "txt":
            return TXTChunker(self.fname)
        elif self.database_name == "pdf":
            return PDFChunker(self.fname)
        elif self.database_name == "docx":
            return DocxChunker(self.fname)
        else:
            raise Exception("Not supported data type.")