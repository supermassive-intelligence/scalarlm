"""
Shared utilities for MMLU-Pro benchmark
"""

import json
import re
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MMLUProError(Exception):
    """Base exception for MMLU-Pro errors"""
    pass


class ScalarLMError(MMLUProError):
    """ScalarLM specific errors"""
    pass


class DatasetError(MMLUProError):
    """Dataset loading/processing errors"""
    pass


class ConfigError(MMLUProError):
    """Configuration related errors"""
    pass


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def safe_json_dump(data: Any, filepath: str, indent: int = 2):
    """Safely dump data to JSON file with NumPy support"""
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=indent)


def safe_json_load(filepath: str) -> Any:
    """Safely load data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ConfigError(f"Failed to load JSON from {filepath}: {e}")


@contextmanager
def handle_evaluation_errors():
    """Context manager for handling evaluation errors"""
    try:
        yield
    except ScalarLMError as e:
        logger.error(f"ScalarLM failed: {e}")
        raise
    except DatasetError as e:
        logger.error(f"Dataset error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise MMLUProError(f"Evaluation failed: {e}") from e


class AnswerExtractor:
    """Shared answer extraction logic for all prompt templates"""
    
    def __init__(self, answer_pattern: str = r"[Tt]he answer is \(?([A-J])\)?"):
        self.answer_pattern = re.compile(answer_pattern, re.IGNORECASE)
    
    def extract_letter(self, response: str, patterns: Optional[List[str]] = None) -> Optional[str]:
        """
        Extract answer letter from model response
        
        Args:
            response: Model response text
            patterns: Additional regex patterns to try
            
        Returns:
            Single letter A-J or None if not found
        """
        if not response:
            return None
            
        # Try primary pattern first
        match = self.answer_pattern.search(response)
        if match:
            answer = match.group(1).upper()
            if answer in 'ABCDEFGHIJ':
                return answer
        
        # Try additional patterns
        if patterns:
            for pattern in patterns:
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                    match = regex.search(response)
                    if match:
                        answer = match.group(1).upper()
                        if answer in 'ABCDEFGHIJ':
                            return answer
                except re.error:
                    continue
        
        # Try "Final Answer: X" pattern
        final_answer_pattern = r"[Ff]inal [Aa]nswer:?\s*\(?([A-J])\)?"
        match = re.search(final_answer_pattern, response)
        if match:
            return match.group(1).upper()
        
        # Fallback: Look for standalone letter
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # Check if line is just a single letter
            if len(line) == 1 and line.upper() in 'ABCDEFGHIJ':
                return line.upper()
            # Check if line starts with a letter followed by period or parenthesis
            if len(line) >= 2 and line[0].upper() in 'ABCDEFGHIJ' and line[1] in '.):':
                return line[0].upper()
        
        return None


def setup_output_directory(output_dir: str) -> Path:
    """Setup and return output directory path"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def format_choices_with_letters(choices: List[str]) -> List[str]:
    """Format choices with letters A-J"""
    formatted = []
    for idx, choice in enumerate(choices):
        letter = chr(ord('A') + idx)
        formatted.append(f"{letter}. {choice}")
    return formatted


def validate_answer_letter(answer: str) -> bool:
    """Validate that answer is a valid letter A-J"""
    return answer and len(answer) == 1 and answer.upper() in 'ABCDEFGHIJ'