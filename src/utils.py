import pandas as pd
from typing_extensions import Optional




def correct_page_cited(row : pd.Series) -> Optional[bool]:
    """
    Checks if the ground truth filename is present in the cited contexts.
    Args:
        row (pd.Series): A row from the evaluation dataset containing 'cited_context' and 'file_name' columns.
    Returns:
        Optional[bool]: True if the ground truth filename is cited, False otherwise. None if no cited contexts.
    """

    if row['cited_context'] is None: return None
    cited_contexts = row['cited_context']
    gt_filename = row['file_name']

    if gt_filename in cited_contexts:
        return True
    else:
        return False


def correct_page_retrieved(row : pd.Series) -> Optional[bool]:
    """
    Checks if the correct page has been retrieved from the RAG system

    Args:
        row (pd.Series): A row from the evaluation dataset containing 'retrieved_context', 'company_symbol', 'report_year', and 'page_number' columns.

    Returns:
        Optional[bool]: True if the correct page is retrieved, False otherwise. None if no retrieved contexts.
    """
    if row['retrieved_context'] is None: return None

    retrieved_contexts = row['retrieved_context']
    company_symbol = row['company_symbol']
    report_year = str(row['report_year'])
    page_number = row['page_number']

    for context in retrieved_contexts:
        if '-' in context.metadata['page_num']:
            start_page, end_page = map(int, context.metadata['page_num'].split('-'))
            expanded_pages = list(range(start_page, end_page + 1))
        else:
            expanded_pages = [int(context.metadata['page_num'])]

        if (
            company_symbol == context.metadata['company_symbol'] 
            and report_year == context.metadata['report_year'] 
            and page_number  in expanded_pages
        ):
            return True
        
    return False
