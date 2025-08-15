"""
MMLU-Pro Evaluator with refactored components
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

from config import MMLUProConfig, EvaluationMode
from dataset import MMLUProDataset, MMLUProQuestion
from prompts import PromptFormatter
from metrics import MMLUProMetrics
from scalarlm_client import ScalarLMClient
from result_handler import ResultHandler
from utils import handle_evaluation_errors, setup_output_directory

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
    Main evaluator class for MMLU-Pro benchmark (Refactored)
    """
    
    def __init__(self, config: MMLUProConfig):
        """
        Initialize evaluator with refactored components
        
        Args:
            config: MMLUProConfig instance
        """
        self.config = config
        
        # Initialize components
        self.dataset = None
        self.prompt_formatter = PromptFormatter(config)
        self.metrics = MMLUProMetrics()
        self.scalarlm_client = None
        self.result_handler = None
        
        # Setup components
        self._init_components()
    
    def _init_components(self):
        """Initialize all components"""
        # Setup output directory
        output_dir = setup_output_directory(self.config.output.output_dir)
        self.result_handler = ResultHandler(str(output_dir))
        
        # Initialize ScalarLM client
        self.scalarlm_client = ScalarLMClient(
            api_url=self.config.scalarlm.api_url,
            model_name=self.config.scalarlm.model_name
        )
        
        # Load dataset
        self.dataset = MMLUProDataset(self.config)
        
        # Log initialization
        stats = self.dataset.get_statistics()
        logger.info(f"Initialized evaluator with {stats['total_questions']} questions from {stats['num_subjects']} subjects")
    
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
        
        # Handle chat format
        if isinstance(prompt, list):  # Chat format returns list of messages
            prompt_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
        else:
            prompt_str = prompt
        
        # Generate response
        generation_params = {
            'max_tokens': self.config.generation.max_tokens,
        }
        
        response, latency = self.scalarlm_client.generate_single(
            prompt_str, 
            **generation_params
        )
        
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
                "evaluation_mode": self.config.evaluation.evaluation_mode.value,
                "num_few_shot": len(examples) if examples else 0
            }
        )
    
    def evaluate_batch(self, questions: List[MMLUProQuestion]) -> List[EvaluationResult]:
        """
        Evaluate a batch of questions
        
        Args:
            questions: List of questions
        
        Returns:
            List of EvaluationResults
        """
        # Format all prompts
        prompts = []
        examples_list = []
        
        for q in questions:
            # Get few-shot examples if needed
            examples = None
            if self.config.evaluation.num_few_shot_examples > 0:
                examples = self.dataset.get_few_shot_examples(
                    q.subject, 
                    n=self.config.evaluation.num_few_shot_examples,
                    exclude_id=q.question_id
                )
            examples_list.append(examples)
            
            prompt = self.prompt_formatter.format_question(q, examples)
            if isinstance(prompt, list):  # Chat format
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
            prompts.append(prompt)
        
        # Generate responses for batch
        generation_params = {
            'max_tokens': self.config.generation.max_tokens,
        }
        
        response_tuples = self.scalarlm_client.generate_batch(prompts, **generation_params)
        
        # Process responses
        results = []
        for q, prompt, examples, (response, latency) in zip(questions, prompts, examples_list, response_tuples):
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
                latency=latency,
                metadata={
                    "num_choices": len(q.choices),
                    "evaluation_mode": self.config.evaluation.evaluation_mode.value,
                    "batch_size": len(questions)
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
        with handle_evaluation_errors():
            logger.info("Starting MMLU-Pro evaluation...")
            logger.info(f"Model: {self.config.scalarlm.model_name or 'default'}")
            logger.info(f"Mode: {self.config.evaluation.evaluation_mode.value}")
            logger.info(f"Subjects: {len(self.dataset.subjects)}")
            
            # Collect all questions from all subjects
            all_questions = []
            subject_question_counts = {}
            
            for subject in self.dataset.subjects:
                questions = self.dataset.get_questions_by_subject(subject)
                all_questions.extend(questions)
                subject_question_counts[subject] = len(questions)
                
                if self.config.output.verbose:
                    logger.info(f"Loaded {subject}: {len(questions)} questions")
            
            logger.info(f"Total questions across all subjects: {len(all_questions)}")
            
            # Evaluate all questions using cross-subject batching
            if self.config.evaluation.batch_size > 1:
                all_results = self._evaluate_cross_subject_batched(all_questions)
            else:
                all_results = self._evaluate_all_sequential(all_questions)
            
            # Group results by subject for metrics calculation
            subject_results = {}
            for subject in self.dataset.subjects:
                subject_results[subject] = [r for r in all_results if r.subject == subject]
                
                # Log progress for each subject
                if self.config.output.verbose and subject_results[subject]:
                    accuracy = sum(r.is_correct for r in subject_results[subject]) / len(subject_results[subject]) * 100
                    logger.info(f"{subject}: {len(subject_results[subject])} questions, {accuracy:.1f}% accuracy")
            
            # Compute metrics
            metrics = self.metrics.compute_metrics(all_results, subject_results)
            
            # Save results
            self.result_handler.save_results(
                all_results, 
                metrics, 
                self.config.to_dict(),
                save_predictions=self.config.output.save_predictions
            )
            
            # Log summary
            self.result_handler.log_summary(
                metrics, 
                save_metrics_per_subject=self.config.output.save_metrics_per_subject
            )
            
            return {
                "results": all_results,
                "metrics": metrics,
                "config": self.config.to_dict()
            }
    
    def _evaluate_cross_subject_batched(self, questions: List[MMLUProQuestion]) -> List[EvaluationResult]:
        """Evaluate all questions using batch processing across subjects"""
        results = []
        batch_size = self.config.evaluation.batch_size
        
        logger.info(f"Processing {len(questions)} questions with batch size {batch_size}")
        
        # Process questions in batches regardless of subject boundaries
        for i in tqdm(range(0, len(questions), batch_size), desc="Batches"):
            batch = questions[i:i + batch_size]
            actual_batch_size = len(batch)
            
            if self.config.output.verbose:
                logger.info(f"Processing batch {i//batch_size + 1}: {actual_batch_size} questions")
            
            batch_results = self.evaluate_batch(batch)
            results.extend(batch_results)
        
        logger.info(f"Completed evaluation of {len(results)} questions")
        return results
    
    def _evaluate_all_sequential(self, questions: List[MMLUProQuestion]) -> List[EvaluationResult]:
        """Evaluate all questions sequentially (one by one)"""
        results = []
        
        for q in tqdm(questions, desc="Questions"):
            # Get few-shot examples if needed
            examples = None
            if self.config.evaluation.num_few_shot_examples > 0:
                examples = self.dataset.get_few_shot_examples(
                    q.subject,
                    n=self.config.evaluation.num_few_shot_examples,
                    exclude_id=q.question_id
                )
            
            result = self.evaluate_question(q, examples)
            results.append(result)
        
        return results
    
    def _evaluate_subject_batched(self, questions: List[MMLUProQuestion]) -> List[EvaluationResult]:
        """Evaluate subject using batch processing (legacy method, kept for compatibility)"""
        results = []
        batch_size = self.config.evaluation.batch_size
        print(f"\n*** Batch size {batch_size}\n")
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_results = self.evaluate_batch(batch)
            results.extend(batch_results)
        print(f"\n*** actual batch size for results {len(results)}\n")
        return results
    
    def _evaluate_subject_sequential(self, questions: List[MMLUProQuestion]) -> List[EvaluationResult]:
        """Evaluate subject sequentially (one by one)"""
        results = []
        
        for q in tqdm(questions, desc="  Questions", leave=False):
            # Get few-shot examples if needed
            examples = None
            if self.config.evaluation.num_few_shot_examples > 0:
                examples = self.dataset.get_few_shot_examples(
                    q.subject,
                    n=self.config.evaluation.num_few_shot_examples,
                    exclude_id=q.question_id
                )
            
            result = self.evaluate_question(q, examples)
            results.append(result)
        
        return results