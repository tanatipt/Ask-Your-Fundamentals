from src.components.schemas import State
from langchain_core.language_models.chat_models import BaseChatModel
from src.prompts.generate_answer import generate_prompt
from langchain_core.prompts.chat import ChatPromptTemplate

def generate_answer(state : State, generator_llm : BaseChatModel) -> State:
    user_input = """Question: {question}\nContext: {context}\nAnswer:"""
    generate_pt = ChatPromptTemplate(
        [
            ('system', generate_prompt),
            ('human', user_input)
        ]
    )

    generate_chain = generate_pt | generator_llm
    response = generate_chain.invoke({"question" : state.question, "context" : state.formatted_docs}).content

    return {"answer" : response}