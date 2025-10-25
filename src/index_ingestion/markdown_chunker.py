from langchain_core.documents import Document
import ast
from src.mapper import get_class
from src.utils import create_chunk, parse_report_path
from dotenv import load_dotenv
import re
load_dotenv()

def clean_text(page_content : str) -> str:
    """
    Cleans the given page content by removing markdown syntax, HTML tags, and unnecessary whitespace.

    Args:
        page_content (str): The raw content of the page.

    Returns:
        str: The cleaned page content.
    """
    page_content = page_content.strip()
    # Replace images ![alt](url) with just alt text
    page_content = re.sub(r"!\[(.*?)\]\(.*?\)", r"\1", page_content)
    # Replace hyperlinks [text](url) with just text
    page_content  = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", page_content)
    # Remove standalone url
    page_content  = re.sub(r"https?://\S+", "", page_content)
    # Remove HTML tags
    page_content  = re.sub(r"<[^>]+>", "", page_content)

    page_content = re.sub(r"\n{3,}", "\n\n", page_content)
    page_content = re.sub(r"[ \t]+", " ", page_content)

    return page_content


class MarkdownChunker:
    """ Class to handle chunking of markdown text using specified chunking method and parameters. """
    def __init__(self, chunker_method : str, chunk_size : int, chunk_overlap: int, chunker_params : dict):
        """
        Initializes the MarkdownChunker with the specified chunking method, size, overlap, and parameters.

        Args:
            chunker_method (str): The method to use for markdown chunking.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
            chunker_params (dict): Additional parameters for the chunker.

        Raises:
            Exception: If the chunking method is invalid.
            Exception: If chunk overlap is greater than or equal to chunk size.
        """

        chunker_cls = get_class('splitter', chunker_method) 
        
        if chunker_cls is None:
            raise Exception('ERROR: Invalid chunking method')
        
        if "headers_to_split_on" in chunker_params:
            chunker_params['headers_to_split_on'] = [ast.literal_eval(item) for item in chunker_params['headers_to_split_on']] 
            self.md_prefixes = tuple(prefix for (prefix, _) in chunker_params['headers_to_split_on'])

        self.chunker = chunker_cls(**chunker_params)

        if chunk_overlap >= chunk_size:
            raise Exception('Chunk overlap cannot be greater or equal to the chunk size')
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, report_chunks : list[Document]) -> list[Document]:
        """
        Merges smaller chunks belonging to the same header section into larger chunks based on the configured chunk size and overlap.

        Args:
            report_chunks (list[Document]): List of document chunks to be merged.

        Returns:
            list[Document]: A list of merged document chunks.
        """
        # List to hold the final merged chunks
        chunks = []
        # A buffer to hold chunks to be merged together
        chunk_buffer = []
        # Variable to track the current size of the chunk being formed
        curr_chunk_size = 0

        # Iterate through each chunk in the report
        for report_chunk in report_chunks:
            # Checking if the current chunk is the start/continuation of a new header section and if the current chunk size exceeds the configured chunk size
            if len(report_chunk.metadata.get('headers')) > 0 and curr_chunk_size > self.chunk_size: 
                # Create a new chunk by merging the buffered chunks
                chunk = create_chunk(buffer = chunk_buffer)
                # Reset the current chunk size for the next chunk
                curr_chunk_size = 0
                overlap_chunks = []

                # If a valid chunk is created, add it to the final chunks list
                if chunk: chunks.append(chunk)

                # Handle the overlap by adding necessary segments from the end of the current chunk buffer
                for chunk_segment in chunk_buffer[::-1]:
                    # Get the length of the current chunk segment
                    chunk_segment_len = len(chunk_segment.page_content)

                    # Check if adding this segment would exceed the overlap size
                    if chunk_segment_len + curr_chunk_size >= self.chunk_overlap:
                        # Calculate the portion of the segment to include for overlap
                        partial_chunk = Document(
                            page_content = chunk_segment.page_content[-(self.chunk_overlap - curr_chunk_size):], 
                            metadata=chunk_segment.metadata
                        )
                        # Add the partial chunk to the overlap chunks and update the current chunk size
                        curr_chunk_size += len(partial_chunk.page_content)
                        overlap_chunks.append(partial_chunk)
                        break
                    else:
                        # Add the entire segment to the overlap chunks and update the current chunk size
                        overlap_chunks.append(chunk_segment)
                        curr_chunk_size += len(chunk_segment.page_content)

                # Reverse the overlap chunks to maintain the original order
                overlap_chunks = overlap_chunks[::-1]
                # Reset the chunk buffer to start forming the next chunk
                chunk_buffer = overlap_chunks

            # Add the current report chunk to the chunk buffer and update the current chunk size
            chunk_buffer.append(report_chunk)
            curr_chunk_size += len(report_chunk.page_content)

        chunk = create_chunk(buffer = chunk_buffer)
        if chunk: chunks.append(chunk)
        return chunks

    def chunk(self, report_pages : list, parsed_file : str) -> list[Document]:
        """
        Chunks the given report pages into smaller segments using the configured chunking method.

        Args:
            report_pages (list): The list of report pages to be chunked.
            parsed_file (str): The path to the parsed file.
        Returns:
            list[Document]: A list of chunked document segments.
        """
        # List to hold all chunks for the report
        report_chunks = []
        # Extract metadata from the parsed file path
        symbol, year, company_name = parse_report_path(parsed_file)

        # Process each parsed page in the report
        for page_data in report_pages:
            # Extract page metadata and cleaning the page content
            page_metadata = page_data['page_metadata']
            page_content = clean_text(page_data['page_content'])


            # Split the page content into chunks using the configured chunker
            page_chunks = self.chunker.split_text(page_content)
            
            # Assign metadata to each chunk of the page
            for chunk in page_chunks:
                chunk.metadata = {"headers" : chunk.metadata,  **page_metadata}
                chunk.page_content = chunk.page_content.strip()

            report_chunks += page_chunks

        # Merge chunks belonging to the same header section to create the list of final chunks
        chunks = self.create_chunks(report_chunks)

        # Add additional metadata to each final chunk
        for chunk in chunks:
            chunk.metadata = {
                "company_name" : company_name, 
                "company_symbol" : symbol, 
                "report_year" : year, 
                "file_path": parsed_file.replace(".json", ".pdf"),
                **chunk.metadata
            }
            chunk.page_content = f"(Company Name: {company_name} / {symbol}, Company Symbol: Report Year: {year}, Page: {chunk.metadata['page_num']})\n{chunk.page_content}"

        return chunks
            