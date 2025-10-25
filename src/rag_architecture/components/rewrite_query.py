from src.rag_architecture.components.schemas import  State
from src.rag_architecture.components.schemas import RewriteOutput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from typing_extensions import List
from langchain_core.messages import trim_messages
from src.rag_architecture.prompts.rewrite_query import rewrite_prompt


def rewrite_query(state : State, rewrite_llm : BaseChatModel , company_info : List) -> State:
    """
    Rewrites the user query based on conversation history and company information, as well
    classifying the user's intention.


    Args:
        state (State): Graph state containing conversation messages.
        rewrite_llm (BaseChatModel): Language model for rewriting the query.
        company_info (List): List of company names and symbols whose information are available.

    Returns:
        State: An updated state with the rewritten user question and user intention.
    """
    # Trim conversation history to select the last 10 messages
    messages = trim_messages(messages = state.messages, token_counter = len,  max_tokens = 10, start_on = "human")
    rewrite_pt = ChatPromptTemplate(
        [
            ('system', rewrite_prompt),
            MessagesPlaceholder('conversation_history')
        ]
    )
    # Rewrite the user query using the language model and prompt template
    rewrite_chain = rewrite_pt | rewrite_llm.with_structured_output(RewriteOutput)
    rewrite_output = rewrite_chain.invoke({"companies" : ",".join(company_info), "conversation_history" : messages})

    # Extract rewritten query and user intention from the output
    rewritten_query = rewrite_output.rewritten_query
    user_intention = rewrite_output.user_intention

    return {"user_question" : rewritten_query, "user_intention" : user_intention}

