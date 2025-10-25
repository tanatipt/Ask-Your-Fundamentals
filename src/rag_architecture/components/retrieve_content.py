from src.rag_architecture.components.schemas import State
from langchain_community.document_transformers import LongContextReorder
from langchain_core.retrievers import BaseRetriever
from src.rag_architecture.components.utils import format_doc
from dotenv import load_dotenv
load_dotenv()

def retrieve_content(state : State, retriever : BaseRetriever) -> State:
    """
    Retrieves relevant documents based on the user's question in the state.

    Args:
        state (State): Graph state containing the user question.
        retriever (BaseRetriever): Retriever to fetch relevant documents.

    Returns:
        State: An updated state with retrieved and formatted documents.
    """
    # Retrieving documents using the retriever
    retrieved_docs = retriever.invoke(state.user_question)
    # Reordering and formatting the retrieved documents to prevent lost in the middle issue
    context_reorder = LongContextReorder()
    reordered_docs = context_reorder.transform_documents(retrieved_docs)
    formatted_docs = format_doc(reordered_docs)

    return {"formatted_docs" : formatted_docs ,"retrieved_docs" : reordered_docs}

