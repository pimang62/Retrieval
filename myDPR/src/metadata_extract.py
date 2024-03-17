import nest_asyncio

nest_asyncio.apply()

import os
import openai

os.environ["OPENAI_API_KEY"] = # ~

from llama_index.llms.openai import OpenAI
from llama_index.core.schema import MetadataMode

llm = OpenAI(temperature=0.1, 
            model='gpt-3.5-turbo', 
            max_tokens=512)

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import TokenTextSplitter

text_spliter = TokenTextSplitter(
    separator=" ",
    chunk_size=512,
    chunk_overlap=128
)

class CustomExtractor(BaseExtractor):
    def extract(self, nodes):
        metadata_list = [
            {
                "custom": (
                    node.metadata["document_title"]
                    + "\n"
                    + node.metadata["excerpt_keywords"]
                )
            }
            for node in nodes
        ]
        return metadata_list

extractors = [
    TitleExtractor(nodes=5, llm=llm)
]

# Chain 만들기
transformations = [text_spliter] + extractors

# Pipeline 짜기
from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(transformations=transformations)

def create_metadata(content: str) -> str:

    """[Document(text=text, extra_info=(?))] 형태로 변형"""
    from llama_index.core import Document
    document = [Document(text=content)]

    nodes = pipeline.run(documents=document)

    """nodes[0].metadata => {"document_title": ~~~}로 나옴"""
    return nodes[0].metadata["document_title"]
