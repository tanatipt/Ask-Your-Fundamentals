from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing_extensions import Literal
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank
from typing_extensions import Any

from dotenv import load_dotenv
load_dotenv()

splitter_map = {
    "MarkdownHeaderTextSplitter" : MarkdownHeaderTextSplitter
}
llm_map = {
    "ChatGoogleGenerativeAI" : ChatGoogleGenerativeAI,
    "ChatOpenAI": ChatOpenAI
}

lexicalstore_map = {
    "BM25Retriever" : BM25Retriever,
    "TFIDFRetriever": TFIDFRetriever
}
vectorstore_map = {
    "Chroma" : Chroma,
    "FAISS" : FAISS
}

embedding_map= {
    "GoogleGenerativeAIEmbeddings" : GoogleGenerativeAIEmbeddings,
    "OpenAIEmbeddings" : OpenAIEmbeddings
}

reranker_map = {
    "CohereRerank" : CohereRerank
}

def get_class(map_type: Literal['splitter', 'llm', 'vectorstore', 'lexicalstore', 'embedding', 'reranker'], name: str) -> Any:
    """
    Retrieves the class corresponding to the given mapping type and name.
    Args:
        map_type (Literal[splitter, llm, vectorstore, lexicalstore, embedding, reranker]): Type of the mapping
        name (str): Name of the class to retrieve

    Raises:
        Exception: Mapping type does not exist
        Exception: Mapping name does not exist in mapping type

    Returns:
        Any: The class corresponding to the specified mapping type and name.
    """
    map_dict = {
        "splitter" : splitter_map, 
        "llm" : llm_map,
        'reranker' : reranker_map,
        'lexicalstore' : lexicalstore_map,
        "vectorstore" : vectorstore_map,
        "embedding" : embedding_map
    }

    if map_type not in map_dict:
        raise Exception('ERROR: Mapping type does not exist')

    map_type_dict = map_dict[map_type]

    if name not in map_type_dict:
        raise Exception('ERROR: Mapping name does not exist in mapping type')
    
    cls = map_type_dict[name]

    return cls
