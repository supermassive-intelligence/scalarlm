"""
Result handling for MMLU-Pro evaluation
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import asdict

from utils import safe_json_dump

logger = logging.getLogger(__name__)


class ResultHandler:
    """Handles saving and logging of evaluation results"""
    
    def __init__(self, output_dir: str):
        """
        Initialize result handler
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_results(self, results: List[Any], metrics: Dict[str, Any], config: Dict[str, Any], 
                    save_predictions: bool = True):
        """
        Save evaluation results to disk
        
        Args:
            results: List of EvaluationResult objects
            metrics: Computed metrics dictionary
            config: Configuration dictionary
            save_predictions: Whether to save detailed predictions
        """
        # Save detailed results
        if save_predictions:
            results_file = self.output_dir / f"results_{self.timestamp}.jsonl"
            with open(results_file, 'w') as f:
                for r in results:
                    f.write(json.dumps(asdict(r)) + '\n')
            logger.info(f"Saved results to {results_file}")
        
        # Save metrics
        metrics_file = self.output_dir / f"metrics_{self.timestamp}.json"
        safe_json_dump(metrics, str(metrics_file))
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save config
        config_file = self.output_dir / f"config_{self.timestamp}.json"
        safe_json_dump(config, str(config_file))
        logger.info(f"Saved config to {config_file}")
    
    def log_summary(self, metrics: Dict[str, Any], save_metrics_per_subject: bool = True):
        """
        Log evaluation summary
        
        Args:
            metrics: Computed metrics
            save_metrics_per_subject: Whether to log per-subject metrics
        """
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        logger.info(f"Total Questions: {metrics['total_questions']}")
        logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
        logger.info(f"Wrong Predictions: {metrics['wrong_predictions']}")
        logger.info(f"No Answer: {metrics['no_answer_count']}")
        logger.info(f"Average Latency: {metrics['avg_latency']:.3f}s")
        logger.info(f"Total Time: {metrics['total_time']:.2f}s")
        
        if save_metrics_per_subject and 'subject_accuracy' in metrics:
            logger.info("\nPer-Subject Accuracy:")
            logger.info("-" * 30)
            # Sort subjects by accuracy
            sorted_subjects = sorted(
                metrics['subject_accuracy'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for subject, accuracy in sorted_subjects:
                counts = metrics['subject_counts'].get(subject, {})
                logger.info(
                    f"{subject:30s}: {accuracy:6.2f}% "
                    f"({counts.get('correct', 0)}/{counts.get('total', 0)})"
                )
        
        # Log domain metrics if available
        if 'domain_metrics' in metrics:
            logger.info("\nDomain-wise Performance:")
            logger.info("-" * 30)
            sorted_domains = sorted(
                metrics['domain_metrics'].items(),
                key=lambda x: x[1]['mean_accuracy'],
                reverse=True
            )
            for domain, domain_stats in sorted_domains:
                logger.info(
                    f"{domain:20s}: {domain_stats['mean_accuracy']:6.2f}% "
                    f"(±{domain_stats['std_accuracy']:.2f})"
                )
    
    def log_progress(self, current: int, total: int, subject: str = "", accuracy: float = None):
        """
        Log evaluation progress
        
        Args:
            current: Current item number
            total: Total items
            subject: Current subject being evaluated
            accuracy: Current accuracy if available
        """
        progress_pct = (current / total) * 100 if total > 0 else 0
        
        if subject and accuracy is not None:
            logger.info(
                f"Progress: {current}/{total} ({progress_pct:.1f}%) - "
                f"{subject} accuracy: {accuracy:.1f}%"
            )
        elif subject:
            logger.info(
                f"Progress: {current}/{total} ({progress_pct:.1f}%) - "
                f"Evaluating {subject}"
            )
        else:
            logger.info(f"Progress: {current}/{total} ({progress_pct:.1f}%)")