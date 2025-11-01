from fastapi import FastAPI
from pydantic import BaseModel
from config import settings
from src.rag_architecture.graph_constructor import GraphConstructor
from langchain_core.messages import convert_to_messages
from typing_extensions import Literal
from langchain_core.messages import convert_to_messages

app = FastAPI()
graph_constructor = GraphConstructor(
    base_input_dir=settings.base_input_dir,
    vectorstore_config=settings.vectorstore_config,
    rerank_config=settings.rerank_config,
    generator_config=settings.generator_config,
    lexicalstore_config=settings.lexicalstore_config,
    ensemble_config=settings.ensemble_config
)
graph = graph_constructor.compile()

class Message(BaseModel):
    role : Literal['user', 'ai']
    content: str

class ChatInput(BaseModel):
    messages: list[Message]

@app.post("/chat/", response_model=tuple)
def chat(chat_input: ChatInput):
    """

    Fast API endpoint to chat with the RAG chatbot

    Args:
        chat_input (ChatInput): Conversation history with the chatbot

    Returns:
        tuple: Chatbot response , citations and classified user intention
    """
    messages = convert_to_messages([message.model_dump() for message in chat_input.messages])

    try:
        response = graph.invoke({"messages" : messages})
        answer = response["answer"]
        user_intention = response["user_intention"]
        citations = response["citations"]
    except Exception as e:
        answer = f"API ERROR : {e}"
        user_intention = None
        citations = []

    return answer, user_intention, citations