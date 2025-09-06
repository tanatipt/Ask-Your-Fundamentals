import nest_asyncio
from typing_extensions import Dict
import fitz
from dotenv import load_dotenv
from src.utils import get_file_paths, path_exists, classify_scanned_pdf
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
import gc
from marker.config.parser import ConfigParser
import torch

load_dotenv()
nest_asyncio.apply()    

class MarkerParser:
    def __init__(self, marker_config : Dict):
        self.marker_config = marker_config

    
    def parse(self, pdf_input_path: str):

        if not path_exists(pdf_input_path):
            raise Exception('ERROR: Input Directory Not Found')
        
        reports = []
        file_paths = get_file_paths(pdf_input_path, '.pdf')
        marker_config = self.marker_config

        for file_path in file_paths:
            md_pages = {}
            document = fitz.open(file_path)
            scanned_page_idx = classify_scanned_pdf(document)
            print('Scanned Idx : ', scanned_page_idx)
            
            for page_idx in range(len(document)):
                converter = md_output = md_metadata = md_content = block_counts = None
                try:
                    print('File Path: ', file_path, ', Page : ', page_idx)
                    if page_idx in scanned_page_idx: 
                        marker_config['disable_ocr'] = False
                    else:
                        marker_config['disable_ocr'] = True

                    page_idx_str = str(page_idx)
                    marker_config['page_range'] = page_idx_str
                    config_parser = ConfigParser(marker_config)

                    converter = PdfConverter(
                        config=config_parser.generate_config_dict(),
                        artifact_dict=create_model_dict(device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
                        processor_list=config_parser.get_processors(),
                        renderer=config_parser.get_renderer(),
                        llm_service=config_parser.get_llm_service()
                    )
                    md_output = converter(file_path)
                    md_metadata = md_output.metadata
                    md_content = md_output.markdown
                    block_counts = {k : v for [k, v] in md_metadata['page_stats'][0]['block_counts']}

                    contain_table = any(tag in block_counts for tag in ['Table', 'TableGroup', 'TableOfContents', 'TableCell'])
                    contain_img = any(tag in block_counts for tag in ['Figure', 'FigureGroup', 'Picture', 'PictureGroup'])

                    md_pages[page_idx + 1] = {
                        "page_metadata" : {"contain_img" : contain_img, "contain_table" : contain_table},
                        "page_content" : md_content
                    }

                except Exception as e:
                    print(f'ERROR: {e}')
                    md_pages[page_idx + 1] = {
                        "page_metadata" : {"contain_img" : False, "contain_table" : False},
                        "page_content" : f'ERROR: {e}'
                    }
                finally:
                    del converter, md_output, md_metadata, md_content, block_counts
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    print('\n')


            report = {'file_path' : file_path, "md_pages" : md_pages}
            reports.append(report)
        return reports


    