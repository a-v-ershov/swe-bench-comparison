"""
Data models
"""

import pandas as pd
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


class TestResult(BaseModel):
    """
    Final output of the test logic
    @param solution_openai: Solution output from OpenAI
    @param solution_deepseek: Solution output from DeepSeek
    @param test_openai: Patch comparison output for OpenAI
    @param test_deepseek: Patch comparison output for DeepSeek
    """

    solution_openai: Solution
    solution_deepseek: Solution
    test_openai: PatchComparisonResponse
    test_deepseek: PatchComparisonResponse

    def create_comparison_record(self, issue: pd.Series) -> dict:
        return {
            # Model names
            "openai_model": self.solution_openai.model_name,
            "deepseek_model": self.solution_deepseek.model_name,
            # Patch solutions
            "openai_patch": self.solution_openai.patch,
            "deepseek_patch": self.solution_deepseek.patch,
            "correct_patch": issue["patch"],
            # Solution descriptions
            "openai_solution_description": self.solution_openai.description,
            "deepseek_solution_description": self.solution_deepseek.description,
            # Number of steps taken
            "openai_num_steps": self.solution_openai.number_of_steps,
            "deepseek_num_steps": self.solution_deepseek.number_of_steps,
            # Number of tokens used for retriever
            "openai_input_tokens_retriever": self.solution_openai.num_input_tokens_for_retriever,
            "deepseek_input_tokens_retriever": self.solution_deepseek.num_input_tokens_for_retriever,
            "openai_output_tokens_retriever": self.solution_openai.num_output_tokens_for_retriever,
            "deepseek_output_tokens_retriever": self.solution_deepseek.num_output_tokens_for_retriever,
            # Number of tokens used for solver
            "openai_input_tokens_solver": self.solution_openai.num_input_tokens_for_solver,
            "deepseek_input_tokens_solver": self.solution_deepseek.num_input_tokens_for_solver,
            "openai_output_tokens_solver": self.solution_openai.num_output_tokens_for_solver,
            "deepseek_output_tokens_solver": self.solution_deepseek.num_output_tokens_for_solver,
            # Comparison
            # Correctness
            "openai_openai_correct": self.test_openai.llm1_is_correct,
            "openai_deepseek_correct": self.test_openai.llm2_is_correct,
            "deepseek_openai_correct": self.test_deepseek.llm1_is_correct,
            "deepseek_deepseek_correct": self.test_deepseek.llm2_is_correct,
            # Scores
            "openai_openai_score": self.test_openai.llm1_score,
            "openai_deepseek_score": self.test_openai.llm2_score,
            "deepseek_openai_score": self.test_deepseek.llm1_score,
            "deepseek_deepseek_score": self.test_deepseek.llm2_score,
            # Descriptions
            "openai_openai_description": self.test_openai.llm1_description,
            "openai_deepseek_description": self.test_openai.llm2_description,
            "deepseek_openai_description": self.test_deepseek.llm1_description,
            "deepseek_deepseek_description": self.test_deepseek.llm2_description,
            # Files
            "openai_openai_files": self.test_openai.llm1_files,
            "openai_deepseek_files": self.test_openai.llm2_files,
            "deepseek_openai_files": self.test_deepseek.llm1_files,
            "deepseek_deepseek_files": self.test_deepseek.llm2_files,
            "correct_patch_files": self.test_openai.files_in_correct_patch,
        }

    @staticmethod
    def get_keys_for_solutions() -> list[str]:
        return [
            "openai_patch",
            "deepseek_patch",
            "correct_patch",
            "openai_solution_description",
            "deepseek_solution_description",
        ]

    @staticmethod
    def get_keys_for_steps() -> list[str]:
        return [
            "openai_num_steps",
            "deepseek_num_steps",
        ]

    @staticmethod
    def get_keys_for_correctness() -> list[str]:
        return [
            "openai_openai_correct",
            "openai_deepseek_correct",
            "deepseek_openai_correct",
            "deepseek_deepseek_correct",
        ]

    @staticmethod
    def get_keys_for_test_scores() -> list[str]:
        return [
            "openai_openai_score",
            "openai_deepseek_score",
            "deepseek_openai_score",
            "deepseek_deepseek_score",
        ]

    @staticmethod
    def get_keys_for_test_descriptions() -> list[str]:
        return [
            "openai_openai_description",
            "openai_deepseek_description",
            "deepseek_openai_description",
            "deepseek_deepseek_description",
        ]

    @staticmethod
    def get_keys_for_files() -> list[str]:
        return [
            "openai_openai_files",
            "openai_deepseek_files",
            "deepseek_openai_files",
            "deepseek_deepseek_files",
            "correct_patch_files",
        ]

    @staticmethod
    def get_keys_for_used_tokens() -> list[str]:
        return [
            "openai_input_tokens_retriever",
            "deepseek_input_tokens_retriever",
            "openai_output_tokens_retriever",
            "deepseek_output_tokens_retriever",
            "openai_input_tokens_solver",
            "deepseek_input_tokens_solver",
            "openai_output_tokens_solver",
            "deepseek_output_tokens_solver",
        ]
