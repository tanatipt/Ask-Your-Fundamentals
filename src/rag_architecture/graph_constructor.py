from src.rag_architecture.components.retrieve_content import retrieve_content
from src.rag_architecture.components.generate_answer import generate_answer
from src.rag_architecture.components.extract_answer import extract_answer
from src.rag_architecture.components.rewrite_query import rewrite_query
from src.rag_architecture.components.generate_response import generate_response
from src.rag_architecture.components.schemas import State, FinalAnswer
from langgraph.graph import StateGraph, START, END
import yfinance as yf
from typing_extensions import Dict
from src.mapper import get_class
import pickle
import os
from langgraph.prebuilt import ToolNode
from langchain.retrievers.ensemble import EnsembleRetriever
from src.rag_architecture.components.utils import calculator, should_continue, route_query
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from dotenv import load_dotenv
load_dotenv()

class GraphConstructor:
    """ Class to construct the RAG architecture graph. """

    def __init__(
        self, 
        base_input_dir : str,
        vectorstore_config : dict, 
        generator_config : dict, 
        lexicalstore_config : dict,
        ensemble_config : dict,
        rerank_config: dict = None,
    ):
        """
        Initializes the GraphConstructor with the specified configurations.

        Args:
            base_input_dir (str): Base directory for input data.
            vectorstore_config (dict): Configuration for the vector store.
            generator_config (dict): Configuration for the generator model.
            lexicalstore_config (dict): Configuration for the lexical store.
            ensemble_config (dict): Configuration for the ensemble retriever.
            rerank_config (dict, optional): Configuration for the reranker. Defaults to None.
        """
        # Defining the tools that the generator model can use
        tools = [calculator, FinalAnswer]

        # Initialize embedding and vectorstore based on the provided configurations
        embedding = get_class('embedding', vectorstore_config.embedding_class)(**vectorstore_config.embedding_params)
        vectorstore = get_class('vectorstore', vectorstore_config.vectorstore_class)(embedding_function = embedding, persist_directory=vectorstore_config.vectorstore_path, **vectorstore_config.vectorstore_params)
        # Initialize retriever from the vectorstore
        vs_retriever = vectorstore.as_retriever(**vectorstore_config.retriever_params)

        self.perform_rerank = rerank_config is not None

        # Initialize the language model to be used
        llm = get_class('llm', generator_config.generator_class)(**generator_config.generator_params)
        # Bind tools to the language model
        llm_w_tools = llm.bind_tools(tools, tool_choice='any', parallel_tool_calls=False)

        # Initialize lexical retriever
        f = open(lexicalstore_config.lexicalstore_path, 'rb')
        lexical_retriever = pickle.load(f)
        lexical_retriever.k = lexicalstore_config.lexicalstore_params.k

        if self.perform_rerank:
            # Initialize reranker and contextual compression retriever after defining the ensemble retriever
            reranker = get_class('reranker', rerank_config.rerank_class)(**rerank_config.rerank_params)
            ensemble_retriever = EnsembleRetriever(retrievers=[lexical_retriever, vs_retriever], weights=[ensemble_config.lexicalstore_weight, ensemble_config.vectorstore_weight])
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=ensemble_retriever
            )
        else:
            # Initialize ensemble retriever without reranking
            retriever = EnsembleRetriever(retrievers=[lexical_retriever, vs_retriever], weights=[ensemble_config.lexicalstore_weight, ensemble_config.vectorstore_weight])

        # Prepare company information for query rewriting and response generation
        report_dir = os.path.join(base_input_dir, 'reports')
        company_symbols = [name for name in os.listdir(report_dir) if os.path.isdir(os.path.join(report_dir, name))]
        company_names = [yf.Ticker(symbol).info.get("longName") for symbol in company_symbols]
        company_info = [f"{company_name} ({company_symbol})" for company_name, company_symbol in zip(company_names, company_symbols)]

        # Initialize nodes in the graph
        self.rewrite_query = self.init_node(rewrite_query, rewrite_llm = llm, company_info=company_info)
        self.retrieve_content =  self.init_node(retrieve_content, retriever = retriever)
        self.generate_answer = self.init_node(generate_answer, generator_llm = llm_w_tools)
        self.extract_answer = self.init_node(extract_answer)
        self.generate_response = self.init_node(generate_response, company_info=company_info)
        self.tool_node = ToolNode(tools=tools)

    def init_node(self, node_function : callable, **kwargs : Dict) -> callable:
        """
        Initializes a node function with additional keyword arguments.
        Args:
            node_function (callable): The node function to be wrapped.

        Returns:
            callable: The wrapped node function with additional arguments.
        """
        def wrapped_node(state : State):
            return node_function(state, **kwargs)
        
        return wrapped_node
    
    def connect_nodes(self) -> StateGraph:
        """        
        Connects the nodes to form the RAG architecture graph.
        Returns:
            StateGraph: The constructed RAG architecture graph.
        """
        workflow = StateGraph(State)

        # Adding nodes to the workflow
        workflow.add_node('rewrite_query', self.rewrite_query)
        workflow.add_node('retrieve_content', self.retrieve_content)
        workflow.add_node('generate_answer', self.generate_answer)
        workflow.add_node('generate_response', self.generate_response)
        workflow.add_node('tools', self.tool_node)
        workflow.add_node('extract_answer', self.extract_answer)

        # Defining edges and conditional flows between nodes
        workflow.add_edge(START, 'rewrite_query')
        workflow.add_conditional_edges('rewrite_query', route_query)
        workflow.add_edge('retrieve_content', 'generate_answer')
        workflow.add_conditional_edges('generate_answer', should_continue)
        workflow.add_edge('tools', 'generate_answer')
        workflow.add_edge('generate_response', END)
        workflow.add_edge('extract_answer', END)

        return workflow
    
    def compile(self, save_path : str = None) -> StateGraph:
        """ Compiles the RAG architecture graph.
        Args:
            save_path (str, optional): Path to save the graph visualization. Defaults to None.
        Returns:
            StateGraph: The compiled RAG architecture graph."""
        workflow = self.connect_nodes()
        graph = workflow.compile()

        if save_path is not None:
            png_graph = graph.get_graph().draw_mermaid_png()

            with open(save_path, "wb") as f:
                f.write(png_graph)

        return graph