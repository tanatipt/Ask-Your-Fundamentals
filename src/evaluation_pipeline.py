from langgraph.graph import StateGraph
import pandas as pd
from src.components.graph_constructor import GraphConstructor
from typing_extensions import Union
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config import settings
from src.components.retrieve_content import format_doc
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm
from langchain_community.callbacks import get_openai_callback
import  time
from dotenv import load_dotenv
load_dotenv()
tqdm.pandas()

class EvaluationPipeline:

    def __init__(
        self, 
        chat_model : Union[StateGraph,  BaseChatModel],
        eval_input_path : str
    ):
        target_cols = [
            'id', 
            'question', 
            'program_answer', 
            'original_answer', 
            'file_name', 
            'company_name', 
            'company_symbol' ,
            'report_year', 
            'page_number', 
            'is_answerable'
        ]

        self.chat_model = chat_model
        eval_dataset = pd.read_csv(eval_input_path)[target_cols]
        self.eval_dataset = eval_dataset

    def generate_answer(self, question : str) -> pd.Series:
        t0 = time.perf_counter()

        if isinstance(self.chat_model, BaseChatModel):
            response = self.chat_model.invoke(question)
            elapsed_time = time.perf_counter() - t0            
            answer = response.content
            token_usage = response.usage_metadata['total_tokens']

            return pd.Series([answer, elapsed_time, token_usage])
        else:
            with get_openai_callback() as cb:
                response = self.chat_model.invoke({"question" : question})
                elapsed_time = time.perf_counter() - t0
                answer = response['answer']
                retrieved_contexts = format_doc(response['retrieved_docs'], include_content=False)

                return pd.Series([answer, retrieved_contexts, elapsed_time,  cb.total_tokens])
        
    def evaluate(self, output_path : str):

        if isinstance(self.chat_model, BaseChatModel):
            new_cols = ['rag_answer', 'elapsed_time', 'token_usage']
        else:
            new_cols = ['rag_answer', 'retrieved_context', 'elapsed_time', 'token_usage']


        self.eval_dataset[new_cols] = self.eval_dataset['question'].progress_apply(self.generate_answer)
        self.eval_dataset.to_csv(output_path, index = False)

    


if __name__ == "__main__":
    graph_constructor = GraphConstructor(
        vectorstore_path=settings.vectorstore_path,
        vectorstore_config=settings.vectorstore_config,
        generator_config=settings.generator_config,
        lexicaldb_path=settings.lexicaldb_path
    )
    graph = graph_constructor.compile()

    #chat_model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', top_p = 0.0, temperature = 0.0)
    #chat_model = ChatOpenAI(model = 'gpt-4.1-mini', temperature = 0.0, top_p = 0.0)
    eval_input_path = 'data/evaluation_qa/qa_dataset_v1.csv'
    eval_output_path = 'results/rag_gemini-2.5-flash_ensemble_chroma0.5_bm250.5.csv'

    eval_pipeline = EvaluationPipeline(chat_model= graph, eval_input_path=eval_input_path)
    eval_pipeline.evaluate(eval_output_path)