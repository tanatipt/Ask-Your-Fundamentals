from src.rag_architecture.components.schemas import State
from typing_extensions import Literal
from langchain_core.documents.base import Document
from sympy import sympify

def route_query(state: State) -> Literal['retrieve_content', 'generate_response']:
    """
    Routes the query based on user intention.
    Args:
        state (State): Graph state containing user intention.

    Returns:
        Literal['retrieve_content', 'generate_response']: The next node to route to.
    """
    user_intention = state.user_intention
    if user_intention == "relevant":
        return "retrieve_content"
    else:
        return "generate_response"
    
def should_continue(state : State) -> Literal['extract_answer', 'tools']:
    """ 
    Decides whether to extract the final answer or invoke tools based on the last message.
    This is used as part of the ReACT loop.

    Args:
        state (State): Graph state containing messages.
    Returns:
        Literal['extract_answer', 'tools']: The next node to route to."""
    messages = state.messages
    last_message = messages[-1]

    if last_message.tool_calls and last_message.tool_calls[0]['name'] == 'FinalAnswer': 
        return "extract_answer"

    return "tools"
    

def format_doc(docs : list[Document], include_content = True) -> str:
    """Format retrieved documents into a string for context.

    Args:
        docs (list[Document]): List of retrieved documents.
        include_content (bool, optional): Flag to include the content or not. Defaults to True.

    Returns:
        str: Formatted string of documents.
    """
    if include_content:
        formatted_doc = "\n\n\n".join(
            [
                f"{'=' * 15} Chunk {i+1} {'=' * 15}\n"
                f"{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )
    else:
        formatted_doc = "\n\n\n".join(
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


def calculator(math_expression : str) -> float:
    """
    A simple calculator that evaluates mathematical expressions. Use this tool whenever you need to calculate an expression—do not evaluate it manually.

    Args:
        math_expression (str): The mathematical expression to evaluate based on your thoughts.

    Raises:
        e: If the expression is invalid or causes an error (e.g., division by zero).

    Returns:
        float: The rounded result of the evaluated mathematical expression.
    """
    try:
        return round(float(sympify(math_expression)), 2)
    except Exception as e:
        raise e


