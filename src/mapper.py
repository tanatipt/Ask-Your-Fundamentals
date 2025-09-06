from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing_extensions import Literal
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

splitter_map = {
    "MarkdownHeaderTextSplitter" : MarkdownHeaderTextSplitter
}
llm_map = {
    "ChatGoogleGenerativeAI" : ChatGoogleGenerativeAI,
    "ChatOpenAI": ChatOpenAI
}

lexicaldb_map = {
    "BM25Retriever" : BM25Retriever
}
vectorstore_map = {
    "Chroma" : Chroma,
    "FAISS" : FAISS
}

embedding_map= {
    "GoogleGenerativeAIEmbeddings" : GoogleGenerativeAIEmbeddings,
    "OpenAIEmbeddings" : OpenAIEmbeddings
}


def get_class(map_type: Literal['splitter', 'llm', 'vectorstore', 'lexicaldb', 'embedding'], name: str):
    map_dict = {
        "splitter" : splitter_map, 
        "llm" : llm_map,
        'lexicaldb' : lexicaldb_map,
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
