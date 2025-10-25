from pymupdf import Page
import fitz
import os
import pandas as pd
from typing_extensions import Optional
from langchain_core.documents import Document
import yfinance as yf
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
punc = string.punctuation

def classify_scanned_pdf(document : list[Page]) -> dict:
    """ 
    Classifies scanned pages in a PDF document.
    Args:
        document (list[Page]): The PDF document represented as a list of pages.

    Returns:
        dict: A dictionary with page indices as keys and Page objects as values for scanned pages.
    """
    scanned_pages = {}

    # Iterate through each page in the document
    for page_idx, page in enumerate(document):
        # Calculate the area covered by images on the page
        img_area = 0.0
        page_area = abs(page.rect)
        searchable_text = page.get_text().strip()

        # Iterate through each image on the page and accumulate their areas
        for img in page.get_image_info(xrefs=True):
            bbox_area = fitz.Rect(img['bbox'])
            img_area += abs(bbox_area)

        # Calculate the percentage of the page covered by images
        img_perc = img_area / page_area

        # If the page has no searchable text and more than 80% image coverage, classify it as scanned
        if not searchable_text and img_perc >= 0.8:
            scanned_pages[page_idx] = page
    
    return scanned_pages


def get_file_paths(input_path : str, file_extension : str) -> list:
    """
    Retrieves all file paths with the specified extension from the given directory or file.

    Args:
        input_path (str): The directory or file path to search.
        file_extension (str): The file extension to filter by.

    Returns:
        list: A list of file paths with the specified extension.
    """
    if os.path.isfile(input_path) : return [input_path]
    file_paths = []

    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(file_extension):
                file_paths.append(os.path.join(root, file))

    return file_paths


def format_page_num(buffer: list[Document]) -> str:
    """
    Formats the page numbers from a list of Document chunks into a concise string representation. 

    Args:
        buffer (list[Document]): A list of Document chunks with metadata containing page numbers.

    Returns:
        str: A formatted string representing the page numbers and ranges.
    """
    page_nums = sorted({c.metadata['page_num'] for c in buffer})
    ranges = []
    start = prev = page_nums[0]

    for n in page_nums[1:]:
        if n == prev + 1:  
            prev = n
        else:  
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n

    ranges.append(f"{start}-{prev}" if start != prev else str(start)) 

    return ",".join(ranges)

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


def create_chunk(buffer: list[Document]) -> Document:
    """
    Creates a single Document chunk by combining the contents of the provided buffer.

    Args:
        buffer (list[Document]): A list of Document chunks to be combined.

    Returns:
        Document: A single Document chunk combining the contents of the buffer.
    """
    # If the buffer is empty, return None
    if len(buffer) == 0:
        return None
    else:
        # Create and return a new Document with combined content and metadata
        return Document(
            metadata={
                'contain_img': any(c.metadata.get('contain_img', False) for c in buffer),
                'contain_table': any(c.metadata.get('contain_table', False) for c in buffer),
                'page_num': format_page_num(buffer)
            },
            page_content='\n'.join(c.page_content for c in buffer)
        )

def parse_report_path(path: str) -> tuple[str, str, str]:
    """
    Parses the report file path to extract the company symbol, report year, and company name.

    Args:
        path (str): The file path to be parsed to extract metadatas

    Returns:
        tuple[str, str, str]: A tuple containing the company symbol, report year, and company name.
    """
    parts = os.path.normpath(path).split(os.sep)
    symbol = parts[-2]  
    year = os.path.splitext(parts[-1])[0] 
    ticker = yf.Ticker(symbol)
    company_name = ticker.info.get("longName")

    return symbol, year, company_name


def preprocess_text(text : str) -> list[str]:
    """
    Preprocesses a text to generate a list of keywords, applying lowercasing/punctuation removal, tokenisation,
    stopword removal and stemming

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        list[str]: A list of preprocessed keywords from the input text.
    """
    # Lowercasing and punctuation removal
    text = text.strip()
    text = text.lower()
    text = text.translate(str.maketrans('', '', punc))

    # Tokenisation
    tokens = word_tokenize(text)
    # Stopword Removal
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    stemmed_unigrams = [stemmer.stem(token) for token in filtered_tokens]
    #stemmed_bigrams = list(" ".join([stemmer.stem(token) for token in n_gram]) for n_gram in ngrams(tokens, 2) if n_gram[0] not in stop_words and n_gram[-1] not in stop_words )
    #stemmed_trigrams = list(" ".join([stemmer.stem(token) for token in n_gram]) for n_gram in ngrams(tokens, 3) if n_gram[0] not in stop_words and n_gram[-1] not in stop_words)

    return stemmed_unigrams #+ stemmed_bigrams + stemmed_trigrams
