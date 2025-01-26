"""
Utility functions for logging and reading credentials
"""

import logging
from typing import TypedDict

import yaml


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with the specified name.
    @param name: Name of the logger
    @return: Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class Credentials(TypedDict):
    openai_key: str
    openrouterai_key: str


def get_credentials(credentials_path: str = "credentials.yaml") -> Credentials:
    """
    Read OpenAI and Anthropic API keys from credentials file.

    @param credentials_path: Path to credentials YAML file
    @return: Tuple of (OpenAI API key, Anthropic API key)
    @raises FileNotFoundError: If credentials file doesn't exist
    @raises KeyError: If required keys are missing from credentials
    """
    try:
        with open(credentials_path, "r") as file:
            credentials = yaml.safe_load(file)
        missing_keys = []
        for key in ["openai_key", "openrouterai_key"]:
            if key not in credentials:
                missing_keys.append(key)
        if missing_keys:
            raise KeyError(f"Missing required fields in credentials file: {', '.join(missing_keys)}")
        return {
            "openai_key": credentials["openai_key"],
            "openrouterai_key": credentials["openrouterai_key"],
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing credentials file: {e}")
