"""
Logic to create solutions for an issue and test them using different models
"""

import asyncio
import json
from typing import Literal

import pandas as pd
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from source.utils.model import PatchComparisonResponse, RetrieverResponse, Solution, SolverResponse
from source.utils.prompt import get_patch_checker_prompt, get_retriever_prompt, get_solver_prompt
from source.utils.repo import (
    LOCAL_SYMPY_REPO_PATH,
    get_all_files,
    get_files_context,
    get_issue_description,
    git_checkout_commit,
)
from source.utils.utils import Credentials, get_logger

logger = get_logger(__name__)


class Tester:
    _DEFAULT_SYSTEM_PROMPT = {"role": "system", "content": "You are a Python software developer."}

    def __init__(self, credentials: Credentials):
        self.openai_client = AsyncOpenAI(api_key=credentials.openai_key)
        self.deepseek_client = AsyncOpenAI(
            api_key=credentials.openrouterai_key, base_url="https://openrouter.ai/api/v1"
        )

    async def create_solution(
        self, files: list[str], issue_description: str, provider: Literal["openai", "deepseek"]
    ) -> Solution:
        """
        Create a solution for the given issue using the specified provider
        @param files: List of files in the repository
        @param issue_description: Description of the issue
        @param provider: Name of the provider ("openai" or "deepseek")
        """
        try:
            return await self._create_solution(files, issue_description, provider)
        except Exception as e:
            logger.error(f"Error in create_solution: {e}")
            model_name = self._get_model_name(provider)
            return Solution(
                model_name=model_name,
                patch="",
                description="Error in creating the solution",
                number_of_steps=0,
                num_input_tokens_for_retriever=0,
                num_output_tokens_for_retriever=0,
                num_input_tokens_for_solver=0,
                num_output_tokens_for_solver=0,
            )

    async def _create_solution(
        self, files: list[str], issue_description: str, provider: Literal["openai", "deepseek"]
    ) -> Solution:
        model_name = self._get_model_name(provider)
        logger.info(f"Creating a solution using {model_name}")

        # Retrieve helpful files
        retriever_resp, retriever_usage = await self._retrieve_files(files, issue_description, provider)

        # Create a solution
        current_step = 0
        is_solved = False
        context = ""
        history = ""
        solver_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        while not is_solved and current_step < 3:
            solver_prompt = get_solver_prompt(
                issue_description=issue_description,
                files=retriever_resp.files,
                context=context,
                history=history,
                current_step=current_step + 1,
            )
            json_resp, current_step_usage = await self._invoke_llm(
                messages=[
                    self._DEFAULT_SYSTEM_PROMPT,
                    {"role": "user", "content": solver_prompt},
                ],
                provider=provider,
            )
            solver_resp_parsed = SolverResponse.model_validate(json_resp)
            for k in solver_usage.keys():
                solver_usage[k] += current_step_usage[k]
            logger.info(f"Got response for the solver at step {current_step}")
            logger.info(f"Usage at step {current_step}: {current_step_usage}")

            # Check if the solver has solved the issue
            is_solved = solver_resp_parsed.patch != ""
            current_step += 1
            context = get_files_context(solver_resp_parsed.files)
            history = solver_resp_parsed
            logger.info(f"Solution at step {current_step}: {solver_resp_parsed.solution}")
            logger.info(f"Patch at step {current_step}: {solver_resp_parsed.patch}")
            logger.info(f"Files at step {current_step}: {solver_resp_parsed.files}")
            logger.info(f"History at step {current_step}: {solver_resp_parsed.history}")

        # Return the solution
        return Solution(
            model_name=model_name,
            patch=solver_resp_parsed.patch,
            description=solver_resp_parsed.solution,
            number_of_steps=current_step,
            num_input_tokens_for_retriever=retriever_usage["prompt_tokens"],
            num_output_tokens_for_retriever=retriever_usage["completion_tokens"],
            num_input_tokens_for_solver=solver_usage["prompt_tokens"],
            num_output_tokens_for_solver=solver_usage["completion_tokens"],
        )

    async def test_solution(
        self,
        issue: pd.Series,
        solution_openai: Solution,
        solution_deepseek: Solution,
        provider: Literal["openai", "deepseek"],
    ):
        """
        Test 2 solutions against the correct patch
        @param issue: Issue to solve
        @param solution_openai: Solution generated using OpenAI
        @param solution_deepseek: Solution generated using DeepSeek
        @param provider: Name of the test provider ("openai" or "deepseek")
        """
        try:
            return await self._test_solution(issue, solution_openai, solution_deepseek, provider)
        except Exception as e:
            logger.error(f"Error in test_solution: {e}")
            return PatchComparisonResponse(
                llm1_score=0,
                llm1_description="Error in testing the solution",
                llm2_score=0,
                llm2_description="Error in testing the solution",
                llm1_files=[],
                llm2_files=[],
                files_in_correct_patch=[],
            )

    async def _test_solution(
        self,
        issue: pd.Series,
        solution_openai: Solution,
        solution_deepseek: Solution,
        provider: Literal["openai", "deepseek"],
    ) -> PatchComparisonResponse:
        model_name = self._get_model_name(provider)
        logger.info(f"Testing the solution using {model_name}")

        # Create the prompt for the patch checker
        issue_description = get_issue_description(issue)
        patch_checker_prompt = get_patch_checker_prompt(
            issue_description=issue_description,
            correct_patch=issue["patch"],
            llm1_patch=solution_openai.patch,
            llm2_patch=solution_deepseek.patch,
        )

        # Get response for the patch checker
        json_resp, usage = await self._invoke_llm(
            messages=[
                self._DEFAULT_SYSTEM_PROMPT,
                {"role": "user", "content": patch_checker_prompt},
            ],
            provider=provider,
        )
        patch_checker_resp_parsed = PatchComparisonResponse.model_validate(json_resp)
        logger.info(f"Got response from {model_name} for the patch checker")
        logger.info(f"Patch checker usage: {usage}")
        return patch_checker_resp_parsed

    def _get_client(self, provider: Literal["openai", "deepseek"]) -> AsyncOpenAI:
        return self.openai_client if provider == "openai" else self.deepseek_client

    def _get_model_name(self, provider: Literal["openai", "deepseek"]) -> str:
        return "gpt-4o-mini" if provider == "openai" else "deepseek/deepseek-chat"

    async def _invoke_llm(
        self, messages: list[dict], provider: Literal["openai", "deepseek"], retry_num: int = 0, retry_limit: int = 3
    ) -> tuple[dict, dict]:
        client = self._get_client(provider)
        model_name = self._get_model_name(provider)
        try:
            resp = await client.chat.completions.create(
                messages=messages,
                model=model_name,
                stream=False,
            )
            json_resp = self._parse_llm_json_response(resp)
            usage = self._get_llm_usage(resp)
        except Exception as e:
            if retry_num < retry_limit:
                logger.error(f"Error in invoking LLM. Retrying... Retry number: {retry_num}. Error: {e}")
                return await self._invoke_llm(messages, provider, retry_num + 1, retry_limit)
            else:
                logger.error(f"Error in invoking LLM. Retry limit reached. Error: {e}")
                raise e
        return json_resp, usage

    @staticmethod
    def _parse_llm_json_response(resp: ChatCompletion) -> dict:
        try:
            return json.loads(resp.choices[0].message.content.replace("```json", "").replace("```", ""))
        except Exception as e:
            logger.error(f"LLM response: {resp}")
            raise e

    @staticmethod
    def _get_llm_usage(resp: ChatCompletion) -> dict:
        usage = resp.usage
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }

    async def _retrieve_files(
        self, files: list[str], issue_description: str, provider: Literal["openai", "deepseek"]
    ) -> tuple[RetrieverResponse, dict]:
        """
        Retrieve relevant files for solving the issue

        @param files: List of files to analyze
        @param issue_description: Description of the issue
        @param provider: Name of the provider ("openai" or "deepseek")
        @return: Tuple of (RetrieverResponse, usage statistics)
        """
        retriever_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        all_retrieved_files = []
        model_name = self._get_model_name(provider)

        # Split files into chunks of 600 to overcome the token limit for DeepSeek (16k tokens)
        file_chunks = [files[i : i + 600] for i in range(0, len(files), 600)]

        async def process_chunk(chunk: list[str]) -> tuple[list[str], dict]:
            retriever_prompt = get_retriever_prompt(issue_description=issue_description, files=chunk)
            json_resp, chunk_usage = await self._invoke_llm(
                messages=[
                    self._DEFAULT_SYSTEM_PROMPT,
                    {"role": "user", "content": retriever_prompt},
                ],
                provider=provider,
            )
            resp_parsed = RetrieverResponse.model_validate(json_resp)
            logger.info(f"Got response from {model_name} for chunk. Files:\n{resp_parsed.files}")
            logger.info(f"Chunk usage: {chunk_usage}")
            return resp_parsed.files, chunk_usage

        # Process all chunks in parallel
        results = await asyncio.gather(*[process_chunk(chunk) for chunk in file_chunks])

        # Aggregate results
        for chunk_files, chunk_usage in results:
            all_retrieved_files.extend(chunk_files)
            for k in retriever_usage.keys():
                retriever_usage[k] += chunk_usage[k]

        final_response = RetrieverResponse(files=sorted(list(set(all_retrieved_files))))
        logger.info(f"Total retrieved files: {len(all_retrieved_files)}")
        logger.info(f"Total retriever usage: {retriever_usage}")
        return final_response, retriever_usage


async def process_issue(issue: pd.Series, tester: Tester) -> dict:
    """
    Process testing process for a given issue
    1. Checkout to the base commit of the issue
    2. Get all files in the repository
    3. Create solutions using GPT-4o-mini and DeepSeek V3
    4. Test the solutions using GPT-4o-mini and DeepSeek V3
    @param issue: Issue to process
    @param tester: Tester object to use for testing and creating solutions
    """
    # Checkout to the base commit of the issue
    commit_id = issue["base_commit"]
    git_checkout_commit(LOCAL_SYMPY_REPO_PATH, commit_id)

    # Get all files in the repository
    all_files = get_all_files()
    logger.info(f"Found {len(all_files)} files in the repository")

    # Create the issue description
    issue_description = get_issue_description(issue)

    # Create solutions in parallel
    solutions = await asyncio.gather(
        tester.create_solution(
            files=all_files,
            issue_description=issue_description,
            provider="openai",
        ),
        tester.create_solution(
            files=all_files,
            issue_description=issue_description,
            provider="deepseek",
        ),
    )
    solution_gpt4, solution_deepseek = solutions
    logger.info(f"Solution using GPT-4o-mini: {solution_gpt4}")
    logger.info(f"Solution using DeepSeek V3: {solution_deepseek}")

    # Test solutions in parallel
    test_results = await asyncio.gather(
        tester.test_solution(
            issue=issue, solution_openai=solution_gpt4, solution_deepseek=solution_deepseek, provider="openai"
        ),
        tester.test_solution(
            issue=issue, solution_openai=solution_gpt4, solution_deepseek=solution_deepseek, provider="deepseek"
        ),
    )
    test_solution_using_gpt4, test_solution_using_deepseek = test_results

    logger.info(f"Test of patches using GPT-4o-mini: {test_solution_using_gpt4}")
    logger.info(f"Test of patches using DeepSeek V3: {test_solution_using_deepseek}")

    # Return the results
    return {
        "solution_gpt4": solution_gpt4,
        "solution_deepseek": solution_deepseek,
        "test_solution_using_gpt4": test_solution_using_gpt4,
        "test_solution_using_deepseek": test_solution_using_deepseek,
    }
