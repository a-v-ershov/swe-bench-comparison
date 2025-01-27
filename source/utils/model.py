"""
Data models
"""

from pydantic import BaseModel


class RetrieverResponse(BaseModel):
    """
    List of useful files from the repository for solving the issue
    @param files: List of files to be used at the solution step
    """

    files: list[str]


class SolverResponse(BaseModel):
    """
    LLM response with a patch solution for the issue
    @param patch: Patch solution for the issue
    @param solution: Description of the solution
    @param files: List of files used in the solution
    @param history: History of steps taken to reach the solution
    """

    patch: str
    solution: str
    files: list[str]
    history: str


class Solution(BaseModel):
    """
    Output of the solution logic
    @param model_name: Name of the model used
    @param patch: Patch solution for the issue
    @param description: Description of the solution
    @param number_of_steps: Number of steps taken to reach the solution
    @param num_input_tokens_for_retriever: Number of input tokens for the retriever
    @param num_output_tokens_for_retriever: Number of output tokens for the retriever
    @param num_input_tokens_for_solver: Number of input tokens for the solver
    @param num_output_tokens_for_solver: Number of output tokens for the solver
    """

    model_name: str
    patch: str
    description: str
    number_of_steps: int
    num_input_tokens_for_retriever: int
    num_output_tokens_for_retriever: int
    num_input_tokens_for_solver: int
    num_output_tokens_for_solver: int


class PatchComparisonResponse(BaseModel):
    """
    Output of the patch comparison prompt
    @param llm1_is_correct: Whether the first LLM's patch is correct
    @param llm1_score: Score of the first LLM's patch solution (1-5)
    @param llm1_description: Description of the first LLM's score
    @param llm2_is_correct: Whether the second LLM's patch is correct
    @param llm2_score: Score of the second LLM's patch solution (1-5)
    @param llm2_description: Description of the second LLM's score
    @param llm1_files: List of files used in the first LLM's solution
    @param llm2_files: List of files used in the second LLM's solution
    @param files_in_correct_patch: List of files in the correct patch
    """

    llm1_is_correct: bool
    llm1_score: int
    llm1_description: str
    llm2_is_correct: bool
    llm2_score: int
    llm2_description: str
    llm1_files: list[str]
    llm2_files: list[str]
    files_in_correct_patch: list[str]
