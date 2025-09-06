from typing_extensions import Dict, List
from langchain_core.documents import Document
import ast
from src.mapper import get_class
from src.utils import create_merged_chunk, parse_report_path
from dotenv import load_dotenv
load_dotenv()


class MarkdownChunker:
    def __init__(self, chunker_method : str, chunk_size : int,  chunker_params : Dict):

        chunker_cls = get_class('splitter', chunker_method) 
        
        if chunker_cls is None:
            raise Exception('ERROR: Invalid chunking method')
        
        if "headers_to_split_on" in chunker_params:
            chunker_params['headers_to_split_on'] = [ast.literal_eval(item) for item in chunker_params['headers_to_split_on']] 
            self.md_prefixes = tuple(prefix for (prefix, _) in chunker_params['headers_to_split_on'])

        self.chunker = chunker_cls(**chunker_params)
        self.chunk_size = chunk_size

    def merge_chunks(self, chunks : List[Document]):
        merged_chunks = []
        chunk_buffer = []
        curr_chunk_size = 0

        for chunk in chunks :

            if chunk.metadata.get('headers') and curr_chunk_size >= self.chunk_size: 
                merged_chunk = create_merged_chunk(chunk_buffer)

                if merged_chunk:
                    merged_chunks.append(merged_chunk)

                chunk_buffer = []
                curr_chunk_size = 0

            chunk_buffer.append(chunk)
            curr_chunk_size += len(chunk.page_content)

        final_chunk = create_merged_chunk(chunk_buffer)
        if final_chunk:
            merged_chunks.append(final_chunk)

        return merged_chunks


    def chunk(self, reports : Dict):
        all_chunks = []

        for report in reports:
            report_chunks = []
            report_path = report['file_path']
            pages = report['md_pages']

            symbol, year, company_name = parse_report_path(report_path)
            

            for page_num, page_data in sorted(pages.items()):
                page_metadata = page_data['page_metadata']
                page_content = page_data['page_content'].strip()

                page_chunks = self.chunker.split_text(page_content)

                for chunk in page_chunks:
                    chunk.metadata = {"headers" : chunk.metadata, "page_num" : page_num, **page_metadata}
                    chunk.page_content = chunk.page_content.strip()

                report_chunks += page_chunks
    
            merged_chunks = self.merge_chunks(report_chunks) 
            
            for chunk in merged_chunks:
                chunk.metadata = {
                    "company_name" : company_name, 
                    "company_symbol" : symbol, 
                    "report_year" : year, 
                    **chunk.metadata
                }

            all_chunks += merged_chunks

        return all_chunks
            