"""
Configuration module for MMLU-Pro benchmark
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class EvaluationMode(Enum):
    """Evaluation modes for MMLU-Pro"""
    DIRECT = "direct"  # Direct answer without reasoning
    COT = "cot"  # Chain-of-thought reasoning
    FEW_SHOT = "few_shot"  # Few-shot learning
    ZERO_SHOT = "zero_shot"  # Zero-shot evaluation


class PromptStyle(Enum):
    """Different prompt formatting styles"""
    STANDARD = "standard"  # Standard QA format
    INSTRUCTION = "instruction"  # Instruction-following format
    CHAT = "chat"  # Chat-based format with system message


@dataclass
class MMLUProConfig:
    """Configuration for MMLU-Pro evaluation"""
    
    # ScalarLM configuration
    # api_url: str = "http://localhost:8000"  # ScalarLM API URL
    api_url: str = "https://llama70b.cray-lm.com/"


    # Model configuration
    model_name: str = ""
    device: str = "cuda"
    dtype: str = "float16"  # float16, float32, bfloat16
    
    # Dataset configuration
    dataset_name: str = "TIGER-Lab/MMLU-Pro"
    dataset_split: str = "test"
    cache_dir: Optional[str] = "./cache/mmlupro"
    subjects: Optional[List[str]] = None  # None means all subjects
    max_samples_per_subject: Optional[int] = None  # For quick testing
    
    # Evaluation configuration
    batch_size: int = 128
    evaluation_mode: EvaluationMode = EvaluationMode.COT
    num_few_shot_examples: int = 5
    prompt_style: PromptStyle = PromptStyle.STANDARD
    
    # Generation configuration
    max_new_tokens: int = 1024
    temperature: float = 0.0  # 0 for deterministic
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    repetition_penalty: float = 1.0
    
    # Chain-of-thought specific
    cot_trigger: str = "Let's think step by step."
    extract_answer_pattern: str = r"[Tt]he answer is \(?([A-J])\)?"
    require_reasoning: bool = True  # Enforce reasoning before answer
    
    # Output configuration
    output_dir: str = "./results/mmlupro"
    save_predictions: bool = True
    save_metrics_per_subject: bool = True
    verbose: bool = True
    log_interval: int = 10  # Log progress every N batches
    
    # Advanced options
    use_chat_template: bool = False
    system_prompt: Optional[str] = None
    seed: int = 42
    tensor_parallel_size: int = 1  # For multi-GPU inference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MMLUProConfig":
        """Create config from dictionary"""
        # Convert string enum values back to Enum types
        if "evaluation_mode" in config_dict and isinstance(config_dict["evaluation_mode"], str):
            config_dict["evaluation_mode"] = EvaluationMode(config_dict["evaluation_mode"])
        if "prompt_style" in config_dict and isinstance(config_dict["prompt_style"], str):
            config_dict["prompt_style"] = PromptStyle(config_dict["prompt_style"])
        return cls(**config_dict)


# Domain/subject mapping for MMLU-Pro
MMLU_PRO_SUBJECTS = {
    "math": ["algebra", "geometry", "calculus", "statistics", "abstract_algebra", "elementary_mathematics"],
    "physics": ["physics", "conceptual_physics", "high_school_physics", "college_physics", "astronomy"],
    "chemistry": ["chemistry", "high_school_chemistry", "college_chemistry", "organic_chemistry"],
    "biology": ["biology", "high_school_biology", "college_biology", "molecular_biology", "genetics"],
    "computer_science": ["computer_science", "computer_security", "machine_learning", "programming"],
    "engineering": ["electrical_engineering", "mechanical_engineering", "civil_engineering"],
    "economics": ["economics", "microeconomics", "macroeconomics", "econometrics"],
    "business": ["business", "management", "accounting", "marketing", "finance"],
    "law": ["law", "jurisprudence", "international_law", "constitutional_law"],
    "medicine": ["medicine", "anatomy", "clinical_knowledge", "medical_genetics", "nutrition"],
    "psychology": ["psychology", "developmental_psychology", "clinical_psychology", "cognitive_psychology"],
    "history": ["history", "world_history", "us_history", "european_history", "prehistory"],
    "philosophy": ["philosophy", "logic", "ethics", "epistemology", "metaphysics"],
    "other": ["sociology", "political_science", "geography", "anthropology", "linguistics"]
}

# Flattened list of all subjects
ALL_SUBJECTS = [subject for subjects in MMLU_PRO_SUBJECTS.values() for subject in subjects]