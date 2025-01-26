"""
Functions to interacts with sympy repository and get the dataset from Hugging Face SWE-Bench.
"""

import os
import subprocess

import pandas as pd

from source.utils.utils import get_logger

HUGGING_FACE_SWE_BENCH_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/data/test-00000-of-00001.parquet"
)
SYMPY_REPO_URL = "https://github.com/sympy/sympy.git"
LOCAL_SYMPY_REPO_PATH = os.path.abspath("./data/sympy")


logger = get_logger(__name__)


def get_sympy_dataset() -> pd.DataFrame:
    """
    Get the dataset for the sympy/sympy repository from the Hugging Face SWE-Bench dataset.
    """
    df = pd.read_parquet(HUGGING_FACE_SWE_BENCH_URL)
    return df[df["repo"] == "sympy/sympy"]


def get_issue_description(series: pd.Series) -> str:
    """
    Create a description of an issue to use in prompts
    """
    return f"Description:\n{series['problem_statement']}\n\nHints:\n{series['hints_text']}"


def git_clone_repo(repo_url: str, destination: str):
    """
    Clone a git repository to the specified destination.

    @param repo_url: URL of the git repository
    @param destination: Path to clone the repository to
    """
    try:
        subprocess.run(["git", "clone", repo_url, destination], check=True)
        logger.info(f"Repository {repo_url} cloned to {destination}")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error while cloning repository: {e}")


def git_checkout_commit(repo_path: str, commit_hash: str):
    """
    Roll back to a specific commit in the given repository.

    @param repo_path: Path to the local repository
    @param commit_hash: Commit hash to check out
    """
    original_path = os.getcwd()
    try:
        os.chdir(repo_path)
        subprocess.run(["git", "checkout", commit_hash], check=True)
        logger.info(f"Checked out to commit {commit_hash}")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error while checking out to commit: {e}")
    finally:
        os.chdir(original_path)


def git_apply_patch(repo_path: str, patch_content: str):
    """
    Apply a patch string to the repository.

    @param repo_path: Path to the local repository
    @param patch_content: Patch content as a string
    """
    try:
        os.chdir(repo_path)
        subprocess.run(["git", "apply"], input=patch_content.encode(), text=False, check=True)
        logger.info("Patch applied")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error while applying patch: {e}")
    finally:
        os.chdir("..")


def get_all_files(
    directory: str = LOCAL_SYMPY_REPO_PATH, exclude_prefixes: list[str] = None, exclude_file_types: list[str] = None
) -> list[str]:
    """
    List all files in the given directory and its subdirectories.

    @param directory: Path to the directory
    @param exclude_prefixes: List of filename prefixes to exclude, default
    is ['.git/', '.ci/', '.github/', '.circleci/']
    @param exclude_file_types: List of file types (extensions) to exclude, default is ['.pyc', '.svg']
    @return: List of file paths
    """
    if exclude_prefixes is None:
        exclude_prefixes = [".git/", ".ci/", ".github/", ".circleci/"]
    if exclude_file_types is None:
        exclude_file_types = [".pyc", ".svg"]

    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file_path.startswith(f"{directory}/{prefix}") for prefix in exclude_prefixes):
                continue
            if any(file.endswith(ext) for ext in exclude_file_types):
                continue
            file_paths.append(os.path.join(root, file))
    # Remove the common prefix from the file paths
    file_paths = [f.replace(f"{LOCAL_SYMPY_REPO_PATH}/", "") for f in file_paths]
    return sorted(file_paths)


def get_files_context(files: list[str]) -> str:
    """
    Create a context string from the content of the files.

    @param files: List of file paths
    @return: Context string
    """
    context = ""
    for file in files:
        if not file.startswith(LOCAL_SYMPY_REPO_PATH):
            file = f"{LOCAL_SYMPY_REPO_PATH}/{file}"
        context += f"File: {file}\n\n"
        with open(file, "r") as f:
            context += f.read() + "\n-----------\n"
    return context
