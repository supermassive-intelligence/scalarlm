"""
MMLU-Pro Evaluator with ScalarLM integration
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import numpy as np

import scalarlm
from config import MMLUProConfig, EvaluationMode
from dataset import MMLUProDataset, MMLUProQuestion
from prompts import PromptFormatter
from metrics import MMLUProMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question_id: str
    subject: str
    question: str
    correct_answer: str
    predicted_answer: Optional[str]
    is_correct: bool
    response: str
    prompt: str
    latency: float
    metadata: Dict[str, Any] = None


class MMLUProEvaluator:
    """
    Main evaluator class for MMLU-Pro benchmark using ScalarLM
    """
    
    def __init__(self, config: MMLUProConfig):
        """
        Initialize evaluator
        
        Args:
            config: MMLUProConfig instance
        """
        self.config = config
        self.dataset = None
        self.prompt_formatter = PromptFormatter(config)
        self.metrics = MMLUProMetrics()
        self.llm = None
        self.results = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ScalarLM
        self._init_scalarlm()
        
        # Load dataset
        self._load_dataset()
    
    def _init_scalarlm(self):
        """Initialize ScalarLM connection"""
        logger.info("Initializing ScalarLM...")
        
        # Set API URL if specified (check env var as fallback)
        import os
        api_url = self.config.api_url or os.getenv("SCALARLM_API_URL", "http://localhost:8000")
        if api_url:
            scalarlm.api_url = api_url
            logger.info(f"ScalarLM API URL set to: {api_url}")
        
        self.llm = scalarlm.SupermassiveIntelligence()
        
        # Check health
        try:
            health_status = self.llm.health()
            logger.info(f"ScalarLM health check: {health_status}")
        except Exception as e:
            logger.error(f"ScalarLM health check failed: {e}")
            logger.error(f"Current API URL: {scalarlm.api_url}")
            logger.error("Please check:")
            logger.error("1. Is the ScalarLM server running?")
            logger.error("2. Is the API URL correct?")
            logger.error("3. Try: curl {}/v1/health".format(api_url))
            raise
    
    def _load_dataset(self):
        """Load MMLU-Pro dataset"""
        logger.info("Loading MMLU-Pro dataset...")
        self.dataset = MMLUProDataset(self.config)
        stats = self.dataset.get_statistics()
        logger.info(f"Loaded {stats['total_questions']} questions from {stats['num_subjects']} subjects")
    
    def generate_response(self, prompt: str, model_name: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate response using ScalarLM
        
        Args:
            prompt: Input prompt
            model_name: Optional model name to use
        
        Returns:
            Tuple of (response, latency)
        """
        start_time = time.time()
        
        # Prepare generate kwargs
        generate_kwargs = {
            "prompts": [prompt]  # ScalarLM expects a list
        }
        
        # Only add model_name if it's explicitly set and not empty
        # Empty string or None means use the server's default
        if model_name or self.config.model_name:
            name = model_name or self.config.model_name
            if name and name.strip():  # Only add if not empty string
                generate_kwargs["model_name"] = name
        
        # Add generation parameters if specified
        if self.config.max_new_tokens:
            generate_kwargs["max_tokens"] = self.config.max_new_tokens
        if self.config.temperature != 0.0:
            generate_kwargs["temperature"] = self.config.temperature
        if self.config.top_p != 1.0:
            generate_kwargs["top_p"] = self.config.top_p
        
        try:
            responses = self.llm.generate(**generate_kwargs)
            response = responses[0] if responses else ""
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            response = ""
        
        latency = time.time() - start_time
        return response, latency
    
    def evaluate_question(self, question: MMLUProQuestion, 
                         examples: Optional[List[MMLUProQuestion]] = None) -> EvaluationResult:
        """
        Evaluate a single question
        
        Args:
            question: Question to evaluate
            examples: Few-shot examples
        
        Returns:
            EvaluationResult
        """
        # Format prompt
        prompt = self.prompt_formatter.format_question(question, examples)
        
        # Handle chat format differently if needed
        if isinstance(prompt, list):  # Chat format returns list of messages
            # Convert to string format for ScalarLM
            # You might need to adjust this based on how ScalarLM handles chat templates
            prompt_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
        else:
            prompt_str = prompt
        
        # Generate response
        response, latency = self.generate_response(prompt_str)
        
        # Extract answer
        predicted_answer = self.prompt_formatter.extract_answer(response)
        
        # Check correctness
        is_correct = predicted_answer == question.correct_answer if predicted_answer else False
        
        return EvaluationResult(
            question_id=question.question_id,
            subject=question.subject,
            question=question.question,
            correct_answer=question.correct_answer,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            response=response,
            prompt=prompt_str,
            latency=latency,
            metadata={
                "num_choices": len(question.choices),
                "evaluation_mode": self.config.evaluation_mode.value,
                "num_few_shot": len(examples) if examples else 0
            }
        )
    
    def evaluate_batch(self, questions: List[MMLUProQuestion], 
                      batch_size: Optional[int] = None) -> List[EvaluationResult]:
        """
        Evaluate a batch of questions
        
        Args:
            questions: List of questions
            batch_size: Batch size for processing
        
        Returns:
            List of EvaluationResults
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            
            # Format all prompts
            prompts = []
            for q in batch:
                # Get few-shot examples if needed
                examples = None
                if self.config.num_few_shot_examples > 0:
                    examples = self.dataset.get_few_shot_examples(
                        q.subject, 
                        n=self.config.num_few_shot_examples,
                        exclude_id=q.question_id
                    )
                
                prompt = self.prompt_formatter.format_question(q, examples)
                if isinstance(prompt, list):  # Chat format
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
                prompts.append(prompt)
            
            # Generate responses for batch
            start_time = time.time()
            try:
                # Prepare kwargs
                gen_kwargs = {"prompts": prompts}
                
                # Only add model_name if it's set and not empty
                if self.config.model_name and self.config.model_name.strip():
                    gen_kwargs["model_name"] = self.config.model_name
                
                responses = self.llm.generate(**gen_kwargs)
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                responses = [""] * len(prompts)
            
            batch_latency = time.time() - start_time
            avg_latency = batch_latency / len(prompts)
            
            # Process responses
            if len(responses) == 0:
                return []
            for q, prompt, response in zip(batch, prompts, responses):
                predicted_answer = self.prompt_formatter.extract_answer(response)
                is_correct = predicted_answer == q.correct_answer if predicted_answer else False
                
                result = EvaluationResult(
                    question_id=q.question_id,
                    subject=q.subject,
                    question=q.question,
                    correct_answer=q.correct_answer,
                    predicted_answer=predicted_answer,
                    is_correct=is_correct,
                    response=response,
                    prompt=prompt,
                    latency=avg_latency,
                    metadata={
                        "num_choices": len(q.choices),
                        "evaluation_mode": self.config.evaluation_mode.value,
                        "batch_size": len(batch)
                    }
                )
                results.append(result)
        
        return results
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run full evaluation on the dataset
        
        Returns:
            Dictionary with evaluation results and metrics
        """
        logger.info("Starting MMLU-Pro evaluation...")
        logger.info(f"Model: {self.config.model_name or 'default'}")
        logger.info(f"Mode: {self.config.evaluation_mode.value}")
        logger.info(f"Subjects: {len(self.dataset.subjects)}")
        
        all_results = []
        subject_results = {}
        
        # Evaluate each subject
        for subject in tqdm(self.dataset.subjects, desc="Subjects"):
            questions = self.dataset.get_questions_by_subject(subject)
            
            if self.config.verbose:
                logger.info(f"Evaluating {subject} ({len(questions)} questions)...")
            
            # Evaluate questions for this subject
            if self.config.batch_size > 1:
                logger.info(f"Evaluating {subject} in batches of {self.config.batch_size}...")
                results = self.evaluate_batch(questions)
            else:
                logger.info(f"Evaluating {subject} one by one...")
                results = []
                for q in tqdm(questions, desc=f"  {subject}", leave=False):
                    # Get few-shot examples if needed
                    examples = None
                    if self.config.num_few_shot_examples > 0:
                        examples = self.dataset.get_few_shot_examples(
                            q.subject,
                            n=self.config.num_few_shot_examples,
                            exclude_id=q.question_id
                        )
                    
                    result = self.evaluate_question(q, examples)
                    results.append(result)
            
            all_results.extend(results)
            subject_results[subject] = results
            
            # Log progress
            if self.config.verbose:
                accuracy = sum(r.is_correct for r in results) / len(results) * 100
                logger.info(f"  {subject} accuracy: {accuracy:.1f}%")
        
        # Compute metrics
        metrics = self.metrics.compute_metrics(all_results, subject_results)
        
        # Save results
        self._save_results(all_results, metrics)
        
        # Log summary
        self._log_summary(metrics)
        
        return {
            "results": all_results,
            "metrics": metrics,
            "config": self.config.to_dict()
        }
    
    def _save_results(self, results: List[EvaluationResult], metrics: Dict[str, Any]):
        """Save evaluation results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Custom JSON encoder for NumPy types
        class NumpyEncoder(json.JSONEncoder):
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
        
        # Save detailed results
        if self.config.save_predictions:
            results_file = self.output_dir / f"results_{timestamp}.jsonl"
            with open(results_file, 'w') as f:
                for r in results:
                    f.write(json.dumps(asdict(r), cls=NumpyEncoder) + '\n')
            logger.info(f"Saved results to {results_file}")
        
        # Save metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save config
        config_file = self.output_dir / f"config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, cls=NumpyEncoder)
    
    def _log_summary(self, metrics: Dict[str, Any]):
        """Log evaluation summary"""
        logger.info("EVALUATION SUMMARY")
        logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        logger.info(f"Total Questions: {metrics['total_questions']}")
        logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
        logger.info(f"Average Latency: {metrics['avg_latency']:.3f}s")
        
        if self.config.save_metrics_per_subject:
            logger.info("\nPer-Subject Accuracy:")
            for subject, acc in metrics['subject_accuracy'].items():
                logger.info(f"  {subject}: {acc:.2f}%")
