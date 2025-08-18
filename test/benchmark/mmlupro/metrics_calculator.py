"""
Metrics calculation for MMLU-Pro evaluation
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from utils import safe_json_dump


class MetricsCalculator:
    """Calculate evaluation metrics from results"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_questions = 0
        self.correct_predictions = 0
        self.subject_scores = defaultdict(lambda: {"correct": 0, "total": 0})
        self.latencies = []
        self.no_answer_count = 0
    
    def compute_accuracy(self, correct: int, total: int) -> float:
        """Compute accuracy percentage"""
        if total == 0:
            return 0.0
        return (correct / total) * 100
    
    def compute_metrics(self, all_results: List[Any], 
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
            "overall_accuracy": self.compute_accuracy(
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
            subject_accuracy[subject] = self.compute_accuracy(
                scores["correct"], scores["total"]
            )
            subject_counts[subject] = {
                "correct": scores["correct"],
                "total": scores["total"],
                "wrong": scores["total"] - scores["correct"]
            }
        
        metrics["subject_accuracy"] = subject_accuracy
        metrics["subject_counts"] = subject_counts
        
        # Compute domain-level metrics
        metrics["domain_metrics"] = self._compute_domain_metrics(subject_accuracy)
        
        # Compute statistical metrics
        metrics["statistics"] = self._compute_statistics(all_results)
        
        return metrics
    
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
        if response_lengths:
            stats["response_length"] = {
                "mean": np.mean(response_lengths),
                "std": np.std(response_lengths),
                "min": np.min(response_lengths),
                "max": np.max(response_lengths),
                "median": np.median(response_lengths)
            }
        else:
            stats["response_length"] = {
                "mean": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "median": 0
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