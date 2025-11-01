from src.rag_architecture.components.schemas import State
from typing_extensions import List

def generate_response(state : State, company_info : List) -> State:
    """
    Generates a response based on the user's intention and company information. This function is 
    used when the user's query is deemed irrelevant, general, vague, or unclear.

    Args:
        state (State): Graph state containing user intention and messages.
        company_info (List): List of available company information for reference.

    Returns:
        State: Updated state with generated response and empty retrieved documents.
    """
    user_intention = state.user_intention

    if user_intention == "irrelevant":
        response = f"Sorry, I don’t have data for that company. I only have informaton about the following companies: { ",".join(company_info)}"
    elif user_intention == "general":
        response = "I’m sorry, I can only help with company or financial analysis questions. I am not designed to respond to casual chat, greetings, or non-financial inquiries."
    elif user_intention == "vague":
        response = f"Please include the company name so I can provide the correct financial details. I have information about the following companies: { ",".join(company_info)}"
    elif user_intention == "unclear":
        response = "I’m having trouble understanding your message. Could you rephrase it more clearly?"

    return {"answer" : response, "retrieved_docs" : [], "formatted_docs" : [], "citations" : []}