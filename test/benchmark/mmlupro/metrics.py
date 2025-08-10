"""
Metrics computation for MMLU-Pro evaluation
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MMLUProMetrics:
    """
    Compute and track metrics for MMLU-Pro evaluation
    """
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_questions = 0
        self.correct_predictions = 0
        self.subject_scores = defaultdict(lambda: {"correct": 0, "total": 0})
        self.latencies = []
        self.no_answer_count = 0
    
    def compute_metrics(self, 
                       all_results: List[Any],
                       subject_results: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Compute comprehensive metrics from evaluation results
        
        Args:
            all_results: List of all EvaluationResult objects
            subject_results: Optional dict mapping subjects to their results
        
        Returns:
            Dictionary containing all metrics
        """
        self.reset()
        
        # Process all results
        for result in all_results:
            self.total_questions += 1
            
            if result.is_correct:
                self.correct_predictions += 1
            
            if result.predicted_answer is None:
                self.no_answer_count += 1
            
            # Track per-subject scores
            self.subject_scores[result.subject]["total"] += 1
            if result.is_correct:
                self.subject_scores[result.subject]["correct"] += 1
            
            # Track latency
            self.latencies.append(result.latency)
        
        # Compute aggregate metrics
        metrics = {
            "total_questions": self.total_questions,
            "correct_predictions": self.correct_predictions,
            "wrong_predictions": self.total_questions - self.correct_predictions,
            "no_answer_count": self.no_answer_count,
            "overall_accuracy": self._compute_accuracy(
                self.correct_predictions, self.total_questions
            ),
            "avg_latency": np.mean(self.latencies) if self.latencies else 0,
            "median_latency": np.median(self.latencies) if self.latencies else 0,
            "min_latency": np.min(self.latencies) if self.latencies else 0,
            "max_latency": np.max(self.latencies) if self.latencies else 0,
            "total_time": np.sum(self.latencies) if self.latencies else 0,
        }
        
        # Compute per-subject metrics
        subject_accuracy = {}
        subject_counts = {}
        for subject, scores in self.subject_scores.items():
            subject_accuracy[subject] = self._compute_accuracy(
                scores["correct"], scores["total"]
            )
            subject_counts[subject] = {
                "correct": scores["correct"],
                "total": scores["total"],
                "wrong": scores["total"] - scores["correct"]
            }
        
        metrics["subject_accuracy"] = subject_accuracy
        metrics["subject_counts"] = subject_counts
        
        # Compute domain-level metrics (group subjects by domain)
        metrics["domain_metrics"] = self._compute_domain_metrics(subject_accuracy)
        
        # Compute statistical metrics
        metrics["statistics"] = self._compute_statistics(all_results)
        
        return metrics
    
    def _compute_accuracy(self, correct: int, total: int) -> float:
        """Compute accuracy percentage"""
        if total == 0:
            return 0.0
        return (correct / total) * 100
    
    def _compute_domain_metrics(self, subject_accuracy: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute metrics grouped by domain (e.g., STEM, humanities)
        """
        from config import MMLU_PRO_SUBJECTS
        
        domain_metrics = {}
        
        for domain, subjects in MMLU_PRO_SUBJECTS.items():
            domain_scores = []
            for subject in subjects:
                if subject in subject_accuracy:
                    domain_scores.append(subject_accuracy[subject])
            
            if domain_scores:
                domain_metrics[domain] = {
                    "mean_accuracy": np.mean(domain_scores),
                    "std_accuracy": np.std(domain_scores),
                    "min_accuracy": np.min(domain_scores),
                    "max_accuracy": np.max(domain_scores),
                    "num_subjects": len(domain_scores)
                }
        
        return domain_metrics
    
    def _compute_statistics(self, results: List[Any]) -> Dict[str, Any]:
        """
        Compute statistical metrics
        """
        stats = {}
        
        # Response length statistics
        response_lengths = [len(r.response) for r in results]
        stats["response_length"] = {
            "mean": np.mean(response_lengths),
            "std": np.std(response_lengths),
            "min": np.min(response_lengths),
            "max": np.max(response_lengths),
            "median": np.median(response_lengths)
        }
        
        # Answer distribution
        answer_dist = defaultdict(int)
        for r in results:
            if r.predicted_answer:
                answer_dist[r.predicted_answer] += 1
        stats["answer_distribution"] = dict(answer_dist)
        
        # Correct answer distribution
        correct_dist = defaultdict(int)
        for r in results:
            correct_dist[r.correct_answer] += 1
        stats["correct_answer_distribution"] = dict(correct_dist)
        
        return stats
    
    def create_report(self, metrics: Dict[str, Any]) -> str:
        """
        Create a formatted text report of metrics
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("="*60)
        lines.append("MMLU-Pro Evaluation Report")
        lines.append("="*60)
        
        # Overall metrics
        lines.append("\n## Overall Performance")
        lines.append(f"Total Questions: {metrics['total_questions']}")
        lines.append(f"Correct Predictions: {metrics['correct_predictions']}")
        lines.append(f"Wrong Predictions: {metrics['wrong_predictions']}")
        lines.append(f"No Answer: {metrics['no_answer_count']}")
        lines.append(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        
        # Timing metrics
        lines.append("\n## Timing Metrics")
        lines.append(f"Total Time: {metrics['total_time']:.2f}s")
        lines.append(f"Average Latency: {metrics['avg_latency']:.3f}s")
        lines.append(f"Median Latency: {metrics['median_latency']:.3f}s")
        lines.append(f"Min Latency: {metrics['min_latency']:.3f}s")
        lines.append(f"Max Latency: {metrics['max_latency']:.3f}s")
        
        # Per-subject accuracy
        lines.append("\n## Subject-wise Accuracy")
        sorted_subjects = sorted(
            metrics['subject_accuracy'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for subject, accuracy in sorted_subjects[:10]:  # Top 10
            count = metrics['subject_counts'][subject]
            lines.append(
                f"{subject:30s}: {accuracy:6.2f}% "
                f"({count['correct']}/{count['total']})"
            )
        
        # Domain metrics
        if "domain_metrics" in metrics:
            lines.append("\n## Domain-wise Performance")
            sorted_domains = sorted(
                metrics['domain_metrics'].items(),
                key=lambda x: x[1]['mean_accuracy'],
                reverse=True
            )
            for domain, domain_stats in sorted_domains:
                lines.append(
                    f"{domain:20s}: {domain_stats['mean_accuracy']:6.2f}% "
                    f"(±{domain_stats['std_accuracy']:.2f})"
                )
        
        # Statistics
        if "statistics" in metrics:
            stats = metrics["statistics"]
            lines.append("\n## Response Statistics")
            lines.append(f"Avg Response Length: {stats['response_length']['mean']:.0f} chars")
            lines.append(f"Response Length Std: {stats['response_length']['std']:.0f}")
            
            # Answer distribution
            if stats.get("answer_distribution"):
                lines.append("\n## Predicted Answer Distribution")
                for letter in "ABCDEFGHIJ":
                    count = stats["answer_distribution"].get(letter, 0)
                    pct = (count / metrics['total_questions']) * 100 if metrics['total_questions'] > 0 else 0
                    lines.append(f"  {letter}: {count:4d} ({pct:5.1f}%)")
        
        lines.append("\n" + "="*60)
        
        return "\n".join(lines)
    
    def save_report(self, metrics: Dict[str, Any], filepath: str):
        """Save metrics report to file"""
        report = self.create_report(metrics)
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {filepath}")