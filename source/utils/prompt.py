"""
Functions to generate prompts
"""


def get_retriever_prompt(
    issue_description: str,
    files: list[str],
) -> str:
    with open("./prompts/retriever.txt", "r") as file:
        prompt_template = file.read()
    files_str = "\n".join(files)
    return prompt_template.format(
        issue_description=issue_description,
        files=files_str,
    )


def get_solver_prompt(
    issue_description: str,
    files: list[str],
    context: str,
    history: str,
) -> str:
    with open("./prompts/solver.txt", "r") as file:
        prompt_template = file.read()
    files_str = "\n".join(files)
    return prompt_template.format(
        issue_description=issue_description,
        files=files_str,
        context=context,
        history=history,
    )


def get_patch_checker_prompt(issue_description: str, correct_patch: str, generated_patch: str) -> str:
    with open("./prompts/patch_checker.txt", "r") as file:
        prompt_template = file.read()
    return prompt_template.format(
        issue_description=issue_description,
        correct_patch=correct_patch,
        generated_patch=generated_patch,
    )
