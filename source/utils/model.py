"""
Data models
"""

from pydantic import BaseModel


class RetrieverResponse(BaseModel):
    files: list[str]


class SolverResponse(BaseModel):
    patch: str
    solution: str
    files: list[str]
    history: str


class PatchCheckerResponse(BaseModel):
    correct: bool
    description: str
    files_in_correct_patch: list[str]
    files_in_generated_patch: list[str]


class Solution(BaseModel):
    patch: str
    solution_description: str
    files: list[str]
    history: str
    number_of_steps: int
    number_of_input_tokens: int
    number_of_output_tokens: int
    model_name: str
