#!/usr/bin/env python3
"""
Main entry point for MMLU-Pro benchmark evaluation with ScalarLM
"""

import argparse
import logging
import json
import sys
from pathlib import Path

from config import MMLUProConfig, ScalarLMConfig, DatasetConfig, EvaluationConfig, GenerationConfig, OutputConfig, AdvancedConfig, EvaluationMode, PromptStyle, ALL_SUBJECTS, MMLU_PRO_SUBJECTS
from evaluator import MMLUProEvaluator
from utils import ConfigError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run MMLU-Pro benchmark evaluation with ScalarLM"
    )
    
    # ScalarLM configuration
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="ScalarLM API URL (default: http://localhost:8000)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Model name to use for evaluation (empty for default)"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        help="Specific subjects to evaluate (default: all)"
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        choices=list(MMLU_PRO_SUBJECTS.keys()),
        help="Evaluate specific domains (e.g., math, physics)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per subject (for testing)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/mmlupro",
        help="Cache directory for dataset"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        choices=["direct", "cot", "few_shot", "zero_shot"],
        default="cot",
        help="Evaluation mode"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=5,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["standard", "instruction", "chat"],
        default="instruction",
        help="Prompt formatting style"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        choices=["test", "validation"],
        default="test",
        help="Which dataset split to use (default: test)"
    )

    # Generation configuration
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0 for deterministic)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/mmlupro",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save all predictions to file"
    )
    parser.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="Don't save per-subject metrics"
    )
    
    # Other options
    parser.add_argument(
        "--config-file",
        type=str,
        help="Load configuration from JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (evaluates only a few questions)"
    )

    return parser.parse_args()


def load_config(args) -> MMLUProConfig:
    """Load configuration from args or file with new structure"""
    
    # Start with config file if provided
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config_dict = json.load(f)
            config = MMLUProConfig.from_dict(config_dict)
            logger.info(f"Loaded config from {args.config_file}")
        except Exception as e:
            raise ConfigError(f"Failed to load config file {args.config_file}: {e}")
    else:
        # Create with new grouped structure
        config = MMLUProConfig()
    
    # Override with command line arguments
    # ScalarLM configuration
    if args.api_url:
        config.scalarlm.api_url = args.api_url
    if args.model_name:
        config.scalarlm.model_name = args.model_name
    
    # Dataset configuration
    if args.subjects:
        config.dataset.subjects = args.subjects
    elif args.domains:
        # Convert domains to subjects
        subjects = []
        for domain in args.domains:
            subjects.extend(MMLU_PRO_SUBJECTS.get(domain, []))
        config.dataset.subjects = subjects
    
    config.dataset.split = args.dataset_split
    
    if args.max_samples:
        config.dataset.max_samples_per_subject = args.max_samples
    if args.cache_dir:
        config.dataset.cache_dir = args.cache_dir
    
    # Evaluation configuration
    config.evaluation.evaluation_mode = EvaluationMode(args.evaluation_mode)
    config.evaluation.num_few_shot_examples = args.num_shots
    config.evaluation.prompt_style = PromptStyle(args.prompt_style)
    config.evaluation.batch_size = args.batch_size
    
    # Generation configuration
    config.generation.max_tokens = args.max_tokens
    config.generation.temperature = args.temperature
    
    # Output configuration
    config.output.output_dir = args.output_dir
    config.output.save_predictions = args.save_predictions
    config.output.save_metrics_per_subject = not args.no_save_metrics
    config.output.verbose = args.verbose
    
    # Debug mode
    if args.debug:
        config.dataset.subjects = ["mathematics"]  # Only eval one subject
        config.dataset.max_samples_per_subject = 5  # Only 5 questions
        logger.info("Debug mode: evaluating limited subset")
    
    return config


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set logging level
    if args.debug or args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args)
    
    # Run evaluation
    logger.info("Starting MMLU-Pro evaluation with ScalarLM")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")
    
    try:
        # Create evaluator
        evaluator = MMLUProEvaluator(config)
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Print summary
        metrics = results['metrics']
        print("EVALUATION COMPLETE")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Correct: {metrics['correct_predictions']}")
        print(f"Wrong: {metrics['wrong_predictions']}")
        print(f"No Answer: {metrics['no_answer_count']}")
        print(f"Average Latency: {metrics['avg_latency']:.3f}s")
        print(f"\nResults saved to: {config.output.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())