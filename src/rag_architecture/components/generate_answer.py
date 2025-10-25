from src.rag_architecture.components.schemas import State
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from src.rag_architecture.prompts.generate_answer import generate_prompt
from langchain_core.prompts.chat import MessagesPlaceholder

def generate_answer(state : State, generator_llm : BaseChatModel) -> State:
    """
    Generates an answer based on the provided state and generator language model.
    Either performs tool calls or generates a final answer directly.
    
    Args:
        state (State): Graph state containing context and messages.
        generator_llm (BaseChatModel): Language model for generating answers.

    Returns:
        State: Updated state with generated messages.
    """
    
    generate_pt = ChatPromptTemplate(
        [
            ('system', generate_prompt),
            MessagesPlaceholder('messages')
        ]
    )


    generate_chain = generate_pt | generator_llm
    response = generate_chain.invoke({"context" : state.formatted_docs, 'messages' : state.messages})
    return {"messages" : [response]}