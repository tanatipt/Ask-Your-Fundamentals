import nest_asyncio
import fitz
from dotenv import load_dotenv
from src.utils import classify_scanned_pdf
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
import gc
from marker.config.parser import ConfigParser
import torch

load_dotenv()
nest_asyncio.apply()    

class MarkerParser:
    """ Class to handle parsing of PDF documents into markdown pages using Marker library. """
    def __init__(self, parser_config : dict):
        """ Initializes the MarkerParser with the specified parser configuration.

        Args:
            parser_config (dict): Configuration for the Marker parser.
        """
        self.marker_config = parser_config
    
    def parse(self, input_path: str) -> list:
        """ Parses the given PDF document and returns a list of markdown pages.

        Args:
            input_path (str): Path to the PDF document to be parsed.

        Returns:
            list: A list of dictionaries containing markdown content and metadata for each page.
        """
        
        marker_config = self.marker_config
        md_pages = []
        # Open the PDF document using fitz
        document = fitz.open(input_path)
        # Classify scanned pages in the PDF document
        scanned_page_idx = classify_scanned_pdf(document)

        # Iterate through each page in the document
        for page_idx in range(len(document)):
            converter = md_output = md_metadata = md_content = block_counts = None

            try:
                # Configure OCR settings based on whether the page is scanned
                if page_idx in scanned_page_idx: 
                    marker_config['disable_ocr'] = False
                else:
                    marker_config['disable_ocr'] = True

                # Set the page range for the current page
                page_idx_str = str(page_idx)
                marker_config['page_range'] = page_idx_str
                config_parser = ConfigParser(marker_config)

                # Initialize the PdfConverter with the specified configuration
                converter = PdfConverter(
                    config=config_parser.generate_config_dict(),
                    artifact_dict=create_model_dict(device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
                    processor_list=config_parser.get_processors(),
                    renderer=config_parser.get_renderer(),
                    llm_service=config_parser.get_llm_service()
                )

                # Convert the current page to markdown
                md_output = converter(input_path)
                md_metadata = md_output.metadata
                md_content = md_output.markdown 

                # Extract block counts from the metadata
                block_counts = {k : v for [k, v] in md_metadata['page_stats'][0]['block_counts']}
                # Determine if the page contains tables or images
                contain_table = any(tag in block_counts for tag in ['Table', 'TableGroup', 'TableOfContents', 'TableCell'])
                contain_img = any(tag in block_counts for tag in ['Figure', 'FigureGroup', 'Picture', 'PictureGroup'])

                # Append the markdown content and metadata for the current page to the list
                md_pages.append({
                    "page_metadata" : {"contain_img" : contain_img, "contain_table" : contain_table, "page_num" : page_idx + 1},
                    "page_content" : md_content
                })

            except Exception as e:
                # In case of an error, log the error and append an error message for the current page
                print(f'ERROR: {e}')
                md_pages.append({
                    "page_metadata" : {"contain_img" : False, "contain_table" : False, "page_num" : page_idx + 1},
                    "page_content" : f'ERROR: {e}'
                })
            finally:
                del converter, md_output, md_metadata, md_content, block_counts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


        return md_pages


    