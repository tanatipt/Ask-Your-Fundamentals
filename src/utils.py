from pymupdf import Page
import fitz
import os
import pickle
from typing_extensions import List
from langchain_core.documents import Document
import yfinance as yf

def classify_scanned_pdf(document : List[Page]):
    scanned_pages = {}

    for page_idx, page in enumerate(document):
        img_area = 0.0
        page_area = abs(page.rect)
        searchable_text = page.get_text().strip()

        for img in page.get_image_info(xrefs=True):
            bbox_area = fitz.Rect(img['bbox'])
            img_area += abs(bbox_area)

        img_perc = img_area / page_area

        if not searchable_text and img_perc >= 0.8:
            scanned_pages[page_idx] = page
    
    return scanned_pages

def load_mds(md_input_path : str):
    file_paths = get_file_paths(md_input_path, '.pkl')
    reports = []

    for path in file_paths:
        with open(path, 'rb') as f:
            report_pkl = pickle.load(f)
            reports.append(report_pkl)

    return reports

def get_file_paths(input_path : str, file_extension : str) -> List:
    if os.path.isfile(input_path) : return [input_path]
    file_paths = []

    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(file_extension):
                file_paths.append(os.path.join(root, file))

    return file_paths

def path_exists(path: str) -> bool :
    return os.path.exists(path)

def create_path(path: str):        
    os.makedirs(path, exist_ok= True)

def format_page_num(buffer: List[Document]) -> str:

    page_nums = sorted({c.metadata['page_num'] for c in buffer})
    ranges = []
    start = prev = page_nums[0]

    for n in page_nums[1:]:
        if n == prev + 1:  
            prev = n
        else:  
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n

    ranges.append(f"{start}-{prev}" if start != prev else str(start))  # last range

    return ",".join(ranges)

def create_merged_chunk(buffer: List[Document]) -> Document:
    if len(buffer) == 0:
        return None
    else:
        return Document(
            metadata={
                'contain_img': any(c.metadata.get('contain_img', False) for c in buffer),
                'contain_table': any(c.metadata.get('contain_table', False) for c in buffer),
                'page_num': format_page_num(buffer),
            },
            page_content='\n'.join(c.page_content for c in buffer)
        )

def parse_report_path(path: str) -> tuple[str, str, str]:
    parts = os.path.normpath(path).split(os.sep)
    symbol = parts[-2]  
    year = os.path.splitext(parts[-1])[0] 
    ticker = yf.Ticker(symbol)
    company_name = ticker.info.get("longName")

    return symbol, year, company_name

