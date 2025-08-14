"""
Configuration module for MMLU-Pro benchmark
"""

from dataclasses import dataclass
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
class ScalarLMConfig:
    """ScalarLM connection and model configuration"""
    api_url: str = "http://localhost:8000"
    model_name: str = ""
    device: str = "cuda"
    dtype: str = "float16"  # float16, float32, bfloat16


@dataclass
class DatasetConfig:
    """Dataset loading and filtering configuration"""
    name: str = "TIGER-Lab/MMLU-Pro"
    split: str = "test"
    cache_dir: str = "./cache/mmlupro"
    subjects: Optional[List[str]] = None  # None means all subjects
    max_samples_per_subject: Optional[int] = None  # For quick testing


@dataclass
class EvaluationConfig:
    """Evaluation behavior configuration"""
    batch_size: int = 128
    evaluation_mode: EvaluationMode = EvaluationMode.COT
    num_few_shot_examples: int = 5
    prompt_style: PromptStyle = PromptStyle.STANDARD
    
    # Chain-of-thought specific
    cot_trigger: str = "Let's think step by step."
    extract_answer_pattern: str = r"[Tt]he answer is \(?([A-J])\)?"
    require_reasoning: bool = True


@dataclass
class GenerationConfig:
    """Model generation parameters"""
    max_new_tokens: int = 1024
    temperature: float = 0.0  # 0 for deterministic
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    repetition_penalty: float = 1.0


@dataclass
class OutputConfig:
    """Output and logging configuration"""
    output_dir: str = "./results/mmlupro"
    save_predictions: bool = True
    save_metrics_per_subject: bool = True
    verbose: bool = True
    log_interval: int = 10  # Log progress every N batches


@dataclass
class AdvancedConfig:
    """Advanced options"""
    use_chat_template: bool = False
    system_prompt: Optional[str] = None
    seed: int = 42
    tensor_parallel_size: int = 1  # For multi-GPU inference


@dataclass
class MMLUProConfig:
    """Main configuration container"""
    scalarlm: ScalarLMConfig = None
    dataset: DatasetConfig = None
    evaluation: EvaluationConfig = None
    generation: GenerationConfig = None
    output: OutputConfig = None
    advanced: AdvancedConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.scalarlm is None:
            self.scalarlm = ScalarLMConfig()
        if self.dataset is None:
            self.dataset = DatasetConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.advanced is None:
            self.advanced = AdvancedConfig()
    
    @classmethod
    def create_simple(cls, **kwargs) -> "MMLUProConfig":
        """Create config with simple flat parameters for backwards compatibility"""
        config = cls()
        
        # Map old flat parameters to new structure
        scalarlm_params = ['api_url', 'model_name', 'device', 'dtype']
        dataset_params = ['dataset_name', 'dataset_split', 'cache_dir', 'subjects', 'max_samples_per_subject']
        evaluation_params = ['batch_size', 'evaluation_mode', 'num_few_shot_examples', 'prompt_style', 
                           'cot_trigger', 'extract_answer_pattern', 'require_reasoning']
        generation_params = ['max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 'repetition_penalty']
        output_params = ['output_dir', 'save_predictions', 'save_metrics_per_subject', 'verbose', 'log_interval']
        advanced_params = ['use_chat_template', 'system_prompt', 'seed', 'tensor_parallel_size']
        
        # Apply parameters to appropriate sub-configs
        for key, value in kwargs.items():
            if key in scalarlm_params:
                # Handle name mapping
                if key == 'dataset_name':
                    setattr(config.dataset, 'name', value)
                elif key == 'dataset_split':
                    setattr(config.dataset, 'split', value)
                else:
                    setattr(config.scalarlm, key, value)
            elif key in dataset_params:
                if key == 'dataset_name':
                    setattr(config.dataset, 'name', value)
                elif key == 'dataset_split':
                    setattr(config.dataset, 'split', value)
                else:
                    setattr(config.dataset, key, value)
            elif key in evaluation_params:
                setattr(config.evaluation, key, value)
            elif key in generation_params:
                setattr(config.generation, key, value)
            elif key in output_params:
                setattr(config.output, key, value)
            elif key in advanced_params:
                setattr(config.advanced, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        def convert_value(value):
            if isinstance(value, Enum):
                return value.value
            elif hasattr(value, '__dict__'):  # dataclass
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            else:
                return value
        
        return {k: convert_value(v) for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MMLUProConfig":
        """Create config from dictionary"""
        # Handle nested structure
        if any(key in config_dict for key in ['scalarlm', 'dataset', 'evaluation', 'generation', 'output', 'advanced']):
            # New nested format
            config = cls()
            
            if 'scalarlm' in config_dict:
                config.scalarlm = ScalarLMConfig(**config_dict['scalarlm'])
            if 'dataset' in config_dict:
                dataset_data = config_dict['dataset'].copy()
                # Handle enum conversion
                config.dataset = DatasetConfig(**dataset_data)
            if 'evaluation' in config_dict:
                eval_data = config_dict['evaluation'].copy()
                if 'evaluation_mode' in eval_data and isinstance(eval_data['evaluation_mode'], str):
                    eval_data['evaluation_mode'] = EvaluationMode(eval_data['evaluation_mode'])
                if 'prompt_style' in eval_data and isinstance(eval_data['prompt_style'], str):
                    eval_data['prompt_style'] = PromptStyle(eval_data['prompt_style'])
                config.evaluation = EvaluationConfig(**eval_data)
            if 'generation' in config_dict:
                config.generation = GenerationConfig(**config_dict['generation'])
            if 'output' in config_dict:
                config.output = OutputConfig(**config_dict['output'])
            if 'advanced' in config_dict:
                config.advanced = AdvancedConfig(**config_dict['advanced'])
            
            return config
        else:
            # Legacy flat format - convert to new structure
            return cls.create_simple(**config_dict)


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