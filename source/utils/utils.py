"""
Utility functions for logging and reading credentials
"""

import logging

import yaml
from pydantic import BaseModel


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with the specified name

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


class Credentials(BaseModel):
    openai_key: str
    openrouterai_key: str


def get_credentials(credentials_path: str = "credentials.yaml") -> Credentials:
    """
    Read OpenAI and OpenRouter API keys from credentials file.

    @param credentials_path: Path to credentials YAML file
    @return: Tuple of (OpenAI API key, OpenRouter API key)
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
        return Credentials(
            openai_key=credentials["openai_key"],
            openrouterai_key=credentials["openrouterai_key"],
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing credentials file: {e}")
