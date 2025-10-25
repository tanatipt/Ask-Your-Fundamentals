from pydantic import BaseModel, Field
from typing_extensions import List, Annotated, Optional, Literal
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(BaseModel):
    """
    A Pydantic schema representing the state of a conversation in a Retrieval-Augmented Generation (RAG) system.
    This state includes the history of messages exchanged, the user's question, documents retrieved to answer the question,
    formatted documents, and the final answer provided by the system.
    """

    messages: Annotated[List[BaseMessage], add_messages] = Field([], title = 'Conversion History', description = 'A list of historical messages between the user and the system.')
    user_question : str = Field("", title="User's Question", description = "The question asked by the user to the RAG system.")
    user_intention: Literal['relevant', 'irrelevant', 'vague', 'unclear', 'general'] = Field('relevant', description = "Intention of the user's question")
    retrieved_docs : List[Document] = Field([], title = "Retrieved Documents", description = "A list of documents retrieved to help answer the user's question")
    formatted_docs : str  = Field("", title = "Formatted Documents", description = "The retrieved documents formatted into a readable string")
    answer: str = Field("", title = "Question's Answer", description = "The response provided by the RAG system to the user's question")
    citations: List[str] = Field([], title = "Answer Citation" , description = "The citations from which the answer to the user question was derived from.")

class RewriteOutput(BaseModel):
    """ Schema for the output of the rewrite query step. """
    rewritten_query : str = Field(..., description = "Rewritten user query that is self-contained and independent of prior conversation history. **DO NOT fix typos, garbled text or incoherent phrasing in any way**. I will penalised you if I catch you fixing  typos, garbled text or incoherent phrasing and imprison you.")
    user_intention: Literal['relevant', 'irrelevant', 'vague', 'unclear', 'general'] = Field(..., description = "User's intention based on the rewritten query. This can either be 'relevant', 'irrelevant', 'vague', 'unclear' or 'general'.")
    
class FinalAnswer(BaseModel):
    """
    A Pydantic schema representing the final response to the user's question. 
    This class should only be used after all necessary calculator tools have 
    been called. A final answer should only be provided if the retrieved 
    context contains sufficient information to address the question.
    """
    
    answer: str = Field(
        ..., 
        description="A comprehensive final answer to the user’s question using all the calculations computed. If the available information in the retrieved context is insufficient to answer the question, clearly state that you don’t know or that there isn’t enough information to respond with confidence."
    )

    citations: Optional[List[int]] = Field(
        ...,
        description="Only include the necessary integer IDs (1-based index) of all chunks that contain the relevant information actually used to derive the answer — no extra or unrelated chunks should be listed. Ignore this field if the there is not enough information in the retrieved context to answer the question.",
    )
