"""
Logic to create solutions for an issue and test them using different models
"""

import json

import pandas as pd
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from source.utils.model import PatchCheckerResponse, RetrieverResponse, Solution, SolverResponse
from source.utils.prompt import get_patch_checker_prompt, get_retriever_prompt, get_solver_prompt
from source.utils.repo import (
    LOCAL_SYMPY_REPO_PATH,
    get_all_files,
    get_files_context,
    get_issue_description,
    git_checkout_commit,
)
from source.utils.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = {"role": "system", "content": "You are a Python software developer."}


def _parse_llm_json_response(resp: ChatCompletion) -> dict:
    return json.loads(resp.choices[0].message.content.replace("```json", "").replace("```", ""))


def _get_llm_usage(resp: ChatCompletion) -> dict:
    usage = resp.usage
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


def create_solution(client: OpenAI, files: list[str], issue_description: str, model_name: str) -> Solution:
    logger.info(f"Creating a solution using {model_name}")

    # Create the prompt for the retriever
    retriever_prompt = get_retriever_prompt(issue_description=issue_description, files=files)

    # Get response for the retriever
    retriever_resp = client.chat.completions.create(
        messages=[
            _DEFAULT_SYSTEM_PROMPT,
            {"role": "user", "content": retriever_prompt},
        ],
        model=model_name,
        stream=False,
    )
    retriever_resp_parsed = RetrieverResponse.model_validate(_parse_llm_json_response(retriever_resp))
    logger.info(f"Got response from {model_name} for the retriever. Files:\n{retriever_resp_parsed.files}")
    usage = _get_llm_usage(retriever_resp)
    logger.info(f"Retriever usage: {usage}")

    # Create a solution
    current_step = 0
    is_solved = False
    context = ""
    history = ""
    while not is_solved and current_step < 5:
        solver_prompt = get_solver_prompt(
            issue_description=issue_description,
            files=retriever_resp_parsed.files,
            context=context,
            history=history,
        )
        solver_resp = client.chat.completions.create(
            messages=[
                _DEFAULT_SYSTEM_PROMPT,
                {"role": "user", "content": solver_prompt},
            ],
            model=model_name,
            stream=False,
        )
        solver_resp_parsed = SolverResponse.model_validate(_parse_llm_json_response(solver_resp))
        current_step_usage = _get_llm_usage(solver_resp)
        for k in usage.keys():
            usage[k] += current_step_usage[k]
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
    return Solution(
        patch=solver_resp_parsed.patch,
        solution_description=solver_resp_parsed.solution,
        files=solver_resp_parsed.files,
        history=solver_resp_parsed.history,
        number_of_steps=current_step,
        number_of_input_tokens=usage["prompt_tokens"],
        number_of_output_tokens=usage["completion_tokens"],
        model_name=model_name,
    )


def test_solution(openai_client: OpenAI, issue: pd.Series, solution: Solution, model_name: str) -> PatchCheckerResponse:
    # Create the prompt for the patch checker
    issue_description = get_issue_description(issue)
    patch_checker_prompt = get_patch_checker_prompt(
        issue_description=issue_description,
        correct_patch=issue["patch"],
        generated_patch=solution.patch,
    )

    # Get response for the patch checker
    patch_checker_resp = openai_client.chat.completions.create(
        messages=[
            _DEFAULT_SYSTEM_PROMPT,
            {"role": "user", "content": patch_checker_prompt},
        ],
        model=model_name,
        stream=False,
    )
    patch_checker_resp_parsed = PatchCheckerResponse.model_validate(_parse_llm_json_response(patch_checker_resp))
    usage = _get_llm_usage(patch_checker_resp)
    logger.info(f"Got response from {model_name} for the patch checker")
    logger.info(f"Patch checker usage: {usage}")
    return patch_checker_resp_parsed


def process_issue(issue: pd.Series, openai_client: OpenAI, deepseek_client: OpenAI):
    # Checkout to the base commit of the issue
    commit_id = issue["base_commit"]
    git_checkout_commit(LOCAL_SYMPY_REPO_PATH, commit_id)

    # Get all files in the repository
    all_files = get_all_files()
    logger.info(f"Found {len(all_files)} files in the repository")

    # Create the issue description
    issue_description = get_issue_description(issue)

    # Create a solution using GPT-4o-mini
    solition_gpt4o_mini = create_solution(
        client=openai_client,
        files=all_files,
        issue_description=issue_description,
        model_name="gpt-4o-mini",
    )
    logger.info(f"Solution using GPT-4o-mini: {solition_gpt4o_mini}")

    # Create a solution using DeepSeek
    solution_deepseek = create_solution(
        client=deepseek_client,
        files=all_files,
        issue_description=issue_description,
        model_name="deepseek/deepseek-chat",
    )
    logger.info(f"Solution using DeepSeek V3: {solution_deepseek}")

    # Test solutions
    check_gpt4_using_gpt4 = test_solution(
        openai_client=openai_client,
        issue=issue,
        solution=solition_gpt4o_mini,
        model_name="gpt-4o-mini",
    )
    logger.info(f"Check solution of GPT-4o-mini using GPT-4o-mini: {check_gpt4_using_gpt4}")
    check_gpt4_using_deepseek = test_solution(
        openai_client=openai_client,
        issue=issue,
        solution=solition_gpt4o_mini,
        model_name="deepseek/deepseek-chat",
    )
    logger.info(f"Check solution of GPT-4o-mini using DeepSeek V3: {check_gpt4_using_deepseek}")
    check_deepseek_using_gpt4 = test_solution(
        openai_client=openai_client,
        issue=issue,
        solution=solution_deepseek,
        model_name="gpt-4o-mini",
    )
    logger.info(f"Check solution of DeepSeek V3 using GPT-4o-mini: {check_deepseek_using_gpt4}")
    check_deepseek_using_deepseek = test_solution(
        openai_client=openai_client,
        issue=issue,
        solution=solution_deepseek,
        model_name="deepseek/deepseek-chat",
    )
    logger.info(f"Check solution of DeepSeek V3 using DeepSeek V3: {check_deepseek_using_deepseek}")

    # Return the results
    return {
        "gpt4o_mini": {
            "solution": solition_gpt4o_mini,
            "test": {
                "gpt4o_mini": check_gpt4_using_gpt4,
                "deepseek": check_gpt4_using_deepseek,
            },
        },
        "deepseek": {
            "solution": solution_deepseek,
            "test": {
                "gpt4o_mini": check_deepseek_using_gpt4,
                "deepseek": check_deepseek_using_deepseek,
            },
        },
    }
