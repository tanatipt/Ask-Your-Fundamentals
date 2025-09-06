from src.components.retrieve_content import retrieve_content
from src.components.generate_answer import generate_answer
from src.components.schemas import State
from langgraph.graph import StateGraph, START, END
from typing_extensions import Dict
from src.mapper import get_class
from langchain.retrievers import EnsembleRetriever
import pickle
from config import settings
from dotenv import load_dotenv
load_dotenv()

class GraphConstructor:

    def __init__(
        self, 
        vectorstore_path : str, 
        vectorstore_config : Dict, 
        generator_config : Dict, 
        lexicaldb_path : str = None
    ):
        embedding = get_class('embedding', vectorstore_config.embedding_class)(**vectorstore_config.embedding_params)
        vectorstore = get_class('vectorstore', vectorstore_config.vectorstore_class)(embedding_function = embedding, persist_directory=vectorstore_path, **vectorstore_config.vectorstore_params)
        llm = get_class('llm', generator_config.generator_class)(**generator_config.generator_params)
        retriever = vectorstore.as_retriever(**vectorstore_config.retriever_params)

        if lexicaldb_path is not None:
            f = open(lexicaldb_path, 'rb')
            lexical_retriever = pickle.load(f)
            lexical_retriever.k = 5
            retriever = EnsembleRetriever(retrievers=[retriever, lexical_retriever], weights=[0.5, 0.5])

        self.retrieve_content = self.init_node(retrieve_content, retriever = retriever)
        self.generate_answer = self.init_node(generate_answer, generator_llm = llm)

    def init_node(self, node_function : callable, **kwargs : Dict) -> callable:
        def wrapped_node(state : State):
            return node_function(state, **kwargs)
        
        return wrapped_node
    
    def connect_nodes(self) -> StateGraph:
        workflow = StateGraph(State)

        workflow.add_node('retrieve_content', self.retrieve_content)
        workflow.add_node('generate_answer', self.generate_answer)
        workflow.add_edge(START, 'retrieve_content')
        workflow.add_edge('retrieve_content', 'generate_answer')
        workflow.add_edge('generate_answer', END)

        return workflow
    
    def compile(self, save_path : str = None) -> StateGraph:
        workflow = self.connect_nodes()
        graph = workflow.compile()

        if save_path is not None:
            png_graph = graph.get_graph().draw_mermaid_png()

            with open(save_path, "wb") as f:
                f.write(png_graph)

        return graph
    

if __name__ == '__main__':
    graph_constructor = GraphConstructor(
        vectorstore_path=settings.vectorstore_path,
        vectorstore_config=settings.vectorstore_config,
        generator_config=settings.generator_config,
        lexicaldb_path=settings.lexicaldb_path
    )
    graph = graph_constructor.compile()
    response = graph.invoke({"question" : "What is the formula of ReLU?"})
    print(response['answer'])