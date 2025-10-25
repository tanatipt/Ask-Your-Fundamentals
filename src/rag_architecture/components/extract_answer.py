from src.rag_architecture.components.schemas import State


def extract_answer(state: State) -> State:
    """
    Extracts the final answer and citations from the state.

    Args:
        state (State): Graph state containing messages and retrieved documents.

    Returns:
        State: Updated state with final answer and citations.
    """
    
    messages = state.messages
    last_tool_msg = messages[-1].tool_calls[0]['args']
    final_answer = last_tool_msg['answer']
    citations = last_tool_msg['citations']
    retrieved_docs = state.retrieved_docs
    cited_docs = set()

    if citations is not None:
        for doc_idx, doc in enumerate(retrieved_docs):

            if doc_idx + 1 in citations:

                if '-' in doc.metadata['page_num']:
                    start_page, end_page = map(int, doc.metadata['page_num'].split('-'))
                    expanded_pages = list(range(start_page, end_page + 1))
                else:
                    expanded_pages = [int(doc.metadata['page_num'])]


                for expanded_page in expanded_pages:
                    doc_name =f"pdf/{doc.metadata['company_symbol']}/{doc.metadata['report_year']}/page_{expanded_page}.pdf"
                    cited_docs.add(doc_name)

    cited_docs = list(cited_docs)

    return {"answer" : final_answer, "citations" : cited_docs}