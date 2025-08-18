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
    
    def _math_expressions_equivalent(self, expr1: str, expr2: str) -> bool:
        """Check if two mathematical expressions are equivalent"""
        # Simple normalization for common cases
        def normalize_expr(expr):
            return (expr.replace(' ', '')
                       .replace('e^-', 'e^{-')
                       .replace('e^{-', 'e^-')
                       .replace('{', '')
                       .replace('}', '')
                       .lower())
        
        return normalize_expr(expr1) == normalize_expr(expr2)
    
    def extract_letter(self, response: str, patterns: Optional[List[str]] = None, choices: Optional[List[str]] = None) -> Optional[str]:
        """
        Extract answer letter from model response
        
        Args:
            response: Model response text
            patterns: Additional regex patterns to try
            choices: List of answer choices to match against (for $\boxed{...} format)
            
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
        
        # Try "Answer: X" pattern (official MMLU-Pro pattern)
        answer_pattern = r'.*[aA]nswer:\s*([A-J])'
        match = re.search(answer_pattern, response)
        if match:
            return match.group(1).upper()
        
        # Try "Final Answer: X" pattern
        final_answer_pattern = r"[Ff]inal [Aa]nswer:?\s*\(?([A-J])\)?"
        match = re.search(final_answer_pattern, response)
        if match:
            return match.group(1).upper()
        
        # Try bold markdown pattern **X) content** (NEW)
        bold_pattern = r'\*\*([A-J])\)\s*[^*]+\*\*'
        match = re.search(bold_pattern, response)
        if match:
            return match.group(1).upper()
        
        # Try "Therefore, the correct answer is:" pattern (NEW)
        conclusion_patterns = [
            r'[Tt]herefore,?\s*the\s*correct\s*answer\s*is:?\s*\*?\*?([A-J])\)?\*?\*?',
            r'[Tt]he\s*correct\s*answer\s*is:?\s*\*?\*?([A-J])\)?\*?\*?',
            r'[Cc]orrect\s*answer\s*is:?\s*\*?\*?([A-J])\)?\*?\*?',
            r'[Tt]he\s*best\s*answer\s*is:?\s*\*?\*?([A-J])\)?\*?\*?',
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()
        
        # Try $\boxed{...}$ pattern with flexible matching (PRIORITIZED - run before detailed explanations)
        boxed_patterns = [
            r'\$\\boxed\{([^}]+)\}\$',   # $\boxed{content}$
            r'\\boxed\{([^}]+)\}',       # \boxed{content}
            r'\$\\boxed\{([A-J])\}\$',   # $\boxed{A}$
            r'boxed\{([^}]+)\}',         # boxed{content} (without backslash)
            r'oxed\{([^}]+)\}',          # Handle corrupted \boxed -> oxed
        ]
        
        for pattern in boxed_patterns:
            match = re.search(pattern, response)
            if match:
                boxed_content = match.group(1).strip()
                
                # If the boxed content is already a letter A-J, return it directly
                if len(boxed_content) == 1 and boxed_content.upper() in 'ABCDEFGHIJ':
                    return boxed_content.upper()
                
                # Try to match to one of the choices
                if choices:
                    # Direct exact match first
                    for i, choice in enumerate(choices):
                        if choice.strip() == boxed_content:
                            return chr(65 + i)  # A, B, C, etc.
                    
                    # Try some common mathematical equivalences
                    normalized_content = boxed_content.replace(' ', '').replace('{', '').replace('}', '').replace('-', '').replace('+', '')
                    for i, choice in enumerate(choices):
                        normalized_choice = choice.strip().replace(' ', '').replace('{', '').replace('}', '').replace('-', '').replace('+', '')
                        if normalized_choice == normalized_content:
                            return chr(65 + i)
                    
                    # Try matching mathematical expressions more flexibly
                    for i, choice in enumerate(choices):
                        # Handle cases like "2 + e^-4" vs "2 + e^{-4}"
                        if self._math_expressions_equivalent(boxed_content, choice.strip()):
                            return chr(65 + i)
        
        # Try to find answers in detailed explanations (MOVED AFTER boxed patterns)
        # Look for patterns like "F) Boycotts, Buycotts, Digital technology, Increased Sales"
        detailed_answer_pattern = r'\b([A-J])\)\s*[A-Za-z][^,\n]*(?:,\s*[A-Za-z][^,\n]*)*'
        matches = re.finditer(detailed_answer_pattern, response)
        for match in matches:
            answer_letter = match.group(1).upper()
            if answer_letter in 'ABCDEFGHIJ':
                # Check if this looks like a complete answer choice (has multiple words/phrases)
                full_text = match.group(0)
                if ',' in full_text or len(full_text.split()) > 3:
                    return answer_letter
        
        # Try more specific patterns for answer extraction
        additional_patterns = [
            r'(?:answer|choice|option):\s*\(?([A-J])\)?',  # "answer: A"
            r'(?:is|are)\s*\(?([A-J])\)?\.?\s*$',         # "is A."
            r'\b([A-J])\)\s*[^a-zA-Z]*$',                 # "A) " at end
            r'^\s*([A-J])\s*$',                           # Just "A" on a line
            r'[Cc]hoice\s+([A-J])\s+is\s+correct',       # "Choice A is correct"
        ]
        
        for pattern in additional_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                answer = match.group(1).upper()
                if answer in 'ABCDEFGHIJ':
                    return answer
        
        # Fallback: Look for standalone letter in last few lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):  # Check last 5 lines
            line = line.strip()
            # Remove markdown formatting
            line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            
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