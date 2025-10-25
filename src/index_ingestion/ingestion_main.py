from config import settings
from src.index_ingestion.markdown_chunker import MarkdownChunker
from src.index_ingestion.marker_parser import MarkerParser
from src.mapper import get_class
from src.utils import get_file_paths, preprocess_text
import pickle
import json
import os
from langchain_core.documents import Document

class IndexIngestion:
    """Class to handle the ingestion process of parsing and chunking financial reports for indexing."""

    def __init__(self, parser_config: dict , chunker_config : dict, base_dir : str):
        """
        Initializes the IndexIngestion with parser and chunker configurations.

        Args:
            parser_config (dict): Configuration for the parser.
            chunker_config (dict): Configuration for the chunker.
            base_dir (str): Base directory containing the reports to be processed.
        """
        self.parser_config = parser_config
        # Initialize the MarkerParser with the provided configuration
        self.parser = MarkerParser(self.parser_config)
        self.chunker_config = chunker_config
        # Initialize the MarkdownChunker with the provided configuration
        self.chunker = MarkdownChunker(
            chunker_method=self.chunker_config.chunker_method, 
            chunk_size=self.chunker_config.chunk_size,
            chunk_overlap=self.chunker_config.chunk_overlap, 
            chunker_params=self.chunker_config.chunker_params
        )

        self.base_dir = base_dir


    def parse(self):
        """ Parses all PDF reports in the specified directory and saves the parsed content as JSON files. """
        report_dir = os.path.join(self.base_dir, 'reports')
        report_files = get_file_paths(report_dir, '.pdf')

        # Iterate through each report file and parse it
        for report_file in report_files:

            dir_name, file_name = os.path.split(report_file)
            base_name, _ = os.path.splitext(file_name)

            # Create new directory path for parsed reports
            new_dir = dir_name.replace('reports', 'parsed_reports')
            parsed_path = os.path.join(new_dir, base_name + '.json')

            # Skip parsing if the parsed file already exists
            if os.path.exists(parsed_path): continue

            # Parsing the report file using the MarkerParser object
            report_pages = self.parser.parse(report_file)
            os.makedirs(new_dir, exist_ok=True)

            # Save the parsed report pages as a JSON file
            with open(parsed_path, 'w') as f:
                json.dump(report_pages, f, indent=4)

    def chunk(self) -> list[Document]:
        """
        Chunks the parsed reports into smaller segments for indexing.
        Returns:
            list[Document]: A list of chunked document segments.
        """
        parsed_dir = os.path.join(self.base_dir, 'parsed_reports')
        parsed_files = get_file_paths(parsed_dir, '.json')
        document_chunks = []

        for parsed_file in parsed_files:
            with open(parsed_file, 'r') as f:
                report_pages = json.load(f) 

            chunks = self.chunker.chunk(report_pages = report_pages, parsed_file = parsed_file)
            document_chunks += chunks

        return document_chunks 
    
 



if __name__ == "__main__":
    # Initialize the IndexIngestion with configurations from settings
    ingestion_job = IndexIngestion(parser_config=settings.parser_config, chunker_config=settings.chunker_config, base_dir = settings.base_input_dir)
    # Run the parsing process
    ingestion_job.parse()
    # Run the chunking process and retrieve the document chunks
    document_chunks = ingestion_job.chunk()

    # Initialize the embedding, vectorstore, and lexicalstore using the specified classes and configurations
    embedding = get_class('embedding', settings.vectorstore_config.embedding_class)(**settings.vectorstore_config.embedding_params)
    vectorstore = get_class('vectorstore', settings.vectorstore_config.vectorstore_class).from_documents(documents=document_chunks, embedding = embedding, persist_directory=settings.vectorstore_config.vectorstore_path, **settings.vectorstore_config.vectorstore_params)
    lexicalstore = get_class('lexicalstore', settings.lexicalstore_config.lexicalstore_class).from_documents(documents = document_chunks,preprocess_func = preprocess_text, **settings.lexicalstore_config.lexicalstore_params)
 
    with open( settings.lexicalstore_config.lexicalstore_path, 'wb') as f:
        pickle.dump(lexicalstore, f)