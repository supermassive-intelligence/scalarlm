"""
Metrics computation for MMLU-Pro evaluation (Simplified)
"""

from typing import Dict, List, Any, Optional
from metrics_calculator import MetricsCalculator
from metrics_reporter import MetricsReporter


class MMLUProMetrics:
    """
    Main metrics class that combines calculation and reporting
    """
    
    def __init__(self):
        """Initialize metrics with calculator and reporter"""
        self.calculator = MetricsCalculator()
        self.reporter = MetricsReporter()
    
    def reset(self):
        """Reset all metrics"""
        self.calculator.reset()
    
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
        return self.calculator.compute_metrics(all_results, subject_results)
    
    def create_report(self, metrics: Dict[str, Any]) -> str:
        """
        Create a formatted text report of metrics
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            Formatted report string
        """
        return self.reporter.create_report(metrics)
    
    def save_report(self, metrics: Dict[str, Any], filepath: str):
        """Save metrics report to file"""
        self.reporter.save_report(metrics, filepath)
    
    def create_summary_table(self, metrics: Dict[str, Any]) -> str:
        """Create a concise summary table"""
        return self.reporter.create_summary_table(metrics)
    
    def log_progress_summary(self, current_subject: str, subject_metrics: Dict[str, Any], 
                           overall_progress: Dict[str, int]):
        """Log progress during evaluation"""
        self.reporter.log_progress_summary(current_subject, subject_metrics, overall_progress)