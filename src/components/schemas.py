from pydantic import BaseModel, Field
from typing_extensions import List
from langchain_core.documents.base import Document

class State(BaseModel):
    question : str = Field("", title="User's Question", description = "The question asked by the user to the RAG system.")
    retrieved_docs : List[Document] = Field([], title = "Retrieved Documents", description = "A list of documents retrieved to help answer the user's question")
    formatted_docs : str  = Field("", title = "Formatted Documents", description = "The retrieved documents formatted into a readable string")
    answer: str = Field("", title = "Question's Answer", description = "The response provided by the RAG system to the user's question")