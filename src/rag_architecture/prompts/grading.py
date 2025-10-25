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