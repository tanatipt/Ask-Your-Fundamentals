import pandas as pd
from src.rag_architecture.graph_constructor import GraphConstructor
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from config import settings
from src.rag_architecture.components.retrieve_content import format_doc
from langchain_core.messages.human import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from langchain_community.callbacks import get_openai_callback
from src.utils import correct_page_retrieved, correct_page_cited
import  time
import time
from dotenv import load_dotenv
load_dotenv()
tqdm.pandas()

from pydantic import BaseModel, Field
from typing_extensions import Literal

class GradeOutput(BaseModel):
    """
    Represents the evaluation result of a generated answer.
    The model assigns a binary score indicating whether the generated
    answer matches either of the ground truth answers.
    """

    answer_correctness: Literal[0, 1] = Field(
        ...,
        description="Indicates if the generated answer matches either ground truth answer (1 = match, 0 = no match)."
    )

grading_prompt = """
You are an expert in answer verification. Given a question, two ground truth answers, and a generated answer, your task is to determine whether the
generated answer matches either ground truth. If the ground truth answers are numerical values, the generated answer must match exactly. 
If they are percentages, minor rounding differences are acceptable. This evaluation should only be performed when the <is_answerable>
 flag is set to True. If <is_answerable> is False, the generated answer must explicitly refuse to answer the question.

Return a score of 1 if it matches and 0 if it does not. I will tip you $2,000 for each correctly verified question, so do your best!


***Examples***

<question>: What is the percentage of GS profits were from oil and gas?
<is_answerable>: True
<ground_truth_answer_1>: 0.2546
<ground_truth_answer_2>: 25%
<generated_answer>: The percentage of GS profits from oil and gas were 25.5%.
<answer_correctness>: 1

<question>: What is the operating margins of GS in 2023 in million dollars?
<is_answerable>: True
<ground_truth_answer_1>: 1,950
<ground_truth_answer_2>: 1,950
<generated_answer>: The operating margins of GS in 2023 was 1,950 million dollars.
<answer_correctness>: 1

<question>: What is the total debt of GS in 2024 in million dollars?
<is_answerable>: True
<ground_truth_answer_1>: 2,025
<ground_truth_answer_2>: 2,025
<generated_answer>: The total debt of GS in 2024 was 2,000 million dollars
<answer_correctness>: 0

<question>: What percentage of GS debt was from foreign sources?
<is_answerable>: True
<ground_truth_answer_1>: 0.305
<ground_truth_answer_2>: 31%
<generated_answer>: The percentage of GS debt from foreign sources were 30%.
<answer_correctness>: 0

<question>: What percentage of JPMG debt was from the private sector?
<is_answerable>: False
<ground_truth_answer_1>: 0.201
<ground_truth_answer_2>: 20%
<generated_answer>: The percentage of JPMG debt from private sector were 20%
<answer_correctness>: 0

<question>: What percentage of JPMG debt was from the public sector?
<is_answerable>: False
<ground_truth_answer_1>: 0.81
<ground_truth_answer_2>: 81%
<generated_answer>: Sorry, I do not have enough information to answer the question.
<answer_correctness>: 1
"""

grading_model = ChatOpenAI(model = 'gpt-4.1-mini', temperature = 0.0, top_p = 0.0)

def grade_answer(row : pd.Series) -> Literal[0,1]:
    """
    Grades the generated answer against the ground truth answers using a grading model.
    Args:
        row (pd.Series): A row from the evaluation dataset containing the question, ground truth answers, and generated answer.
    Returns:
        Literal[0,1]: 1 if the generated answer is correct, 0 otherwise"""
    input_msg = """<question>: {question}\n<is_answerable>: {answerable}\n<ground_truth_answer_1>: {gt_1}\n<ground_truth_answer_2>: {gt_2}\n<generated_answer>: {generated_answer}\n<answer_correctness>:"""
    grading_pt = ChatPromptTemplate(
        [
            ('system', grading_prompt),
            ('human', input_msg)
        ]
    )

    grading_chain = grading_pt | grading_model.with_structured_output(GradeOutput)
    answer_correctness = grading_chain.invoke(
        {
            "question" : row['question'], 
            'answerable': row['is_answerable'],
            'gt_1': row['program_answer'], 
            'gt_2' : row['original_answer'], 
            'generated_answer' : row['rag_answer']
        }).answer_correctness
    
    return answer_correctness

class EvaluationPipeline:
    """ Class to handle the evaluation pipeline for RAG-generated answers. """

    def __init__(
        self, 
        chat_model : BaseChatModel,
        eval_input_path : str
    ):
        """
        Initializes the EvaluationPipeline with the specified chat model and evaluation dataset.

        Args:
            chat_model (BaseChatModel): The chat model to use for generating answers.
            eval_input_path (str): Path to the evaluation dataset CSV file.
        """
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

    def generate_answer(self, conversation_history: list) -> pd.Series:
        """
        Generates an answer using the chat model based on the provided conversation history.

        Args:
            conversation_history (list): List of messages representing the conversation history.

        Returns:
            pd.Series: A series containing the generated answer, retrieved contexts, cited contexts, elapsed time, and token usage.
        """

        time.sleep(20)
        t0 = time.perf_counter()
        with get_openai_callback() as cb:
            try:
                response = self.chat_model.invoke({"messages" : conversation_history})
                elapsed_time = time.perf_counter() - t0

                answer = response['answer']
                retrieved_contexts = response['retrieved_docs'] if len(response['retrieved_docs']) > 0 else None 
                cited_contexts = response['citations'] if len(response['citations']) > 0 else None
            except Exception as e:
                elapsed_time = None
                answer = f"ERROR: {e}"
                retrieved_contexts = None
                cited_contexts = None


            return [answer, retrieved_contexts, cited_contexts, elapsed_time,  cb.total_tokens]
        
    def evaluate(self, output_path : str):
        """
        Evaluate the RAG-generated answers and save the results to the specified output path.

        Args:
            output_path (str): Path to save the evaluation results CSV file.
        """
 
        new_cols = ['rag_answer', 'retrieved_context', 'cited_context', 'elapsed_time', 'token_usage']
        self.eval_dataset[new_cols] = None
        conversation_history = []

        # Iterate through each row in the evaluation dataset
        for index, row in tqdm(self.eval_dataset.iterrows(), total=len(self.eval_dataset)):
            question = HumanMessage(content =  row['question'])
            conversation_history.append(question)
            results = self.generate_answer(conversation_history)

            for col , val in zip(new_cols, results):
                self.eval_dataset.at[index, col] = val
            
            conversation_history.append(AIMessage(content = self.eval_dataset.at[index, 'rag_answer']))

        # Compute evaluation metrics
        self.eval_dataset['correct_page_retrieved'] = self.eval_dataset.apply(correct_page_retrieved, axis = 1)
        self.eval_dataset['correct_page_cited'] = self.eval_dataset.apply(correct_page_cited, axis = 1)
        self.eval_dataset['retrieved_context'] = self.eval_dataset['retrieved_context'].apply(lambda x : format_doc(x, include_content=False) if x is not None else x)
        self.eval_dataset['answer_correctness'] = self.eval_dataset.progress_apply(grade_answer, axis = 1)
        self.eval_dataset.to_csv(output_path, index = False)

    


if __name__ == "__main__":
    # Construct the RAG graph model
    graph_constructor = GraphConstructor(
        base_input_dir=settings.base_input_dir,
        vectorstore_config=settings.vectorstore_config,
        rerank_config=settings.rerank_config,
        generator_config=settings.generator_config,
        lexicalstore_config=settings.lexicalstore_config,
        ensemble_config=settings.ensemble_config
    )
    graph = graph_constructor.compile()

    eval_input_path = 'data/evaluation_qa/qa_dataset_v1.csv'
    eval_output_path = 'results/gemini-2.5-flash_final_final.csv'
    # Initialize and run the evaluation pipeline
    eval_pipeline = EvaluationPipeline(chat_model= graph, eval_input_path=eval_input_path)
    eval_pipeline.evaluate(eval_output_path)