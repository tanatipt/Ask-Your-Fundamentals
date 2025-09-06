from src.components.schemas import State
from typing_extensions import List
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever

def format_doc(docs : List[Document], include_content = True):
    if include_content:
        formatted_doc = "\n\n".join(
            [
                f"--- Chunk {i+1} ---\n"
                f"Company Name: {doc.metadata['company_name']}\n"
                f"Company Symbol: {doc.metadata['company_symbol']}\n"
                f"Report Year: {doc.metadata['report_year']}\n"
                f"Page: {doc.metadata['page_num']}\n"
                f"Content: {doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )
    else:
        formatted_doc = "\n\n".join(
            [
                f"--- Chunk {i+1} ---\n"
                f"Company Name: {doc.metadata['company_name']}\n"
                f"Company Symbol: {doc.metadata['company_symbol']}\n"
                f"Report Year: {doc.metadata['report_year']}\n"
                f"Page: {doc.metadata['page_num']}\n"
                for i, doc in enumerate(docs)
            ]
        )


    return formatted_doc

def retrieve_content(state : State, retriever : VectorStoreRetriever) -> State:
    retrieved_docs = retriever.invoke(input = state.question)
    formatted_docs = format_doc(retrieved_docs)
    return {"formatted_docs" : formatted_docs ,"retrieved_docs" : retrieved_docs}