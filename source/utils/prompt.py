"""
Functions to generate prompts
"""


def get_retriever_prompt(issue_description: str, files: list[str]) -> str:
    """
    Creates prompt for the retriever step (find useful files among all files in the repo to solve the issue)
    @param issue_description: Description of the issue
    @param files: List of files in the repository
    """
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
    current_step: int,
) -> str:
    """
    Creates prompt for the solver step (generate a patch solution for the issue)
    @param issue_description: Description of the issue
    @param files: List of useful files from the repository for solving the issue
    @param context: Context of the issue
    @param history: History of steps taken to reach the solution
    @param current_step: Current step in the solution process
    """
    with open("./prompts/solver.txt", "r") as file:
        prompt_template = file.read()
    files_str = "\n".join(files)
    return prompt_template.format(
        issue_description=issue_description,
        files=files_str,
        context=context,
        history=history,
        current_step=current_step,
    )


def get_patch_checker_prompt(issue_description: str, correct_patch: str, llm1_patch: str, llm2_patch: str) -> str:
    """
    Create a prompt for comparing two LLM patch solutions against the correct patch.

    @param issue_description: Description of the issue
    @param correct_patch: The correct patch solution
    @param llm1_patch: First LLM's patch solution
    @param llm2_patch: Second LLM's patch solution
    @return: Formatted prompt string
    """
    with open("./prompts/patch_checker.txt", "r") as file:
        prompt_template = file.read()
    return prompt_template.format(
        issue_description=issue_description,
        correct_patch=correct_patch,
        llm1_patch=llm1_patch,
        llm2_patch=llm2_patch,
    )
