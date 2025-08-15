"""
Metrics reporting for MMLU-Pro evaluation
"""

import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsReporter:
    """Create formatted reports from computed metrics"""
    
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
        if 'subject_accuracy' in metrics:
            lines.append("\n## Subject-wise Accuracy")
            sorted_subjects = sorted(
                metrics['subject_accuracy'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for subject, accuracy in sorted_subjects[:15]:  # Top 15
                count = metrics['subject_counts'][subject]
                lines.append(
                    f"{subject:30s}: {accuracy:6.2f}% "
                    f"({count['correct']}/{count['total']})"
                )
            
            if len(sorted_subjects) > 15:
                lines.append(f"... and {len(sorted_subjects) - 15} more subjects")
        
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
                    f"(±{domain_stats['std_accuracy']:.2f}) "
                    f"[{domain_stats['num_subjects']} subjects]"
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
                total_answers = sum(stats["answer_distribution"].values())
                for letter in "ABCDEFGHIJ":
                    count = stats["answer_distribution"].get(letter, 0)
                    pct = (count / total_answers) * 100 if total_answers > 0 else 0
                    lines.append(f"  {letter}: {count:4d} ({pct:5.1f}%)")
        
        lines.append("\n" + "="*60)
        
        return "\n".join(lines)
    
    def save_report(self, metrics: Dict[str, Any], filepath: str):
        """Save metrics report to file"""
        report = self.create_report(metrics)
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {filepath}")
    
    def create_summary_table(self, metrics: Dict[str, Any]) -> str:
        """Create a concise summary table"""
        lines = []
        lines.append("MMLU-Pro Evaluation Summary")
        lines.append("-" * 40)
        lines.append(f"Overall Accuracy:    {metrics['overall_accuracy']:6.2f}%")
        lines.append(f"Total Questions:     {metrics['total_questions']:6d}")
        lines.append(f"Correct Answers:     {metrics['correct_predictions']:6d}")
        lines.append(f"Average Latency:     {metrics['avg_latency']:6.3f}s")
        lines.append(f"Total Time:          {metrics['total_time']:6.1f}s")
        
        if 'domain_metrics' in metrics:
            lines.append("\nTop Domain Performances:")
            sorted_domains = sorted(
                metrics['domain_metrics'].items(),
                key=lambda x: x[1]['mean_accuracy'],
                reverse=True
            )[:5]  # Top 5
            for domain, stats in sorted_domains:
                lines.append(f"  {domain:15s}: {stats['mean_accuracy']:5.1f}%")
        
        return "\n".join(lines)
    
    def log_progress_summary(self, current_subject: str, subject_metrics: Dict[str, Any], 
                           overall_progress: Dict[str, int]):
        """Log progress during evaluation"""
        accuracy = subject_metrics.get('accuracy', 0)
        total = subject_metrics.get('total', 0)
        
        logger.info(
            f"Completed {current_subject}: {accuracy:.1f}% "
            f"({subject_metrics.get('correct', 0)}/{total}) - "
            f"Overall: {overall_progress['completed']}/{overall_progress['total']} subjects"
        )