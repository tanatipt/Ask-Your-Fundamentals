from config import settings
from src.markdown_chunker import MarkdownChunker
from src.marker_parser import MarkerParser
from src.mapper import get_class
import pickle

if __name__ == "__main__":
    parser = MarkerParser(settings.parser_config)
    chunker = MarkdownChunker(
        chunker_method=settings.chunker_config.chunker_method, 
        chunk_size=settings.chunker_config.chunk_size, 
        chunker_params=settings.chunker_config.chunker_params
    )
    
    reports = parser.parse(pdf_input_path= settings.pdf_path)
    print('Number of Report : ', len(reports))
    chunks = chunker.chunk(reports = reports)
    print('Number of Chunks : ', len(chunks))

    embedding = get_class('embedding', settings.vectorstore_config.embedding_class)(**settings.vectorstore_config.embedding_params)
    vector_db = get_class('vectorstore', settings.vectorstore_config.vectorstore_class).from_documents(documents=chunks, embedding = embedding, persist_directory=settings.vectorstore_path, **settings.vectorstore_config.vectorstore_params)
    bm25_retriever = get_class('lexicaldb', settings.lexicaldb_config.lexicaldb_class).from_documents(documents = chunks, **settings.lexicaldb_config.lexicaldb_params)
 
    with open(settings.lexicaldb_path, 'wb') as f:
        pickle.dump(bm25_retriever, f)