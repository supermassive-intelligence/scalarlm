"""
Dataset module for MMLU-Pro benchmark
Handles loading, caching, and preprocessing of MMLU-Pro data
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

from utils import DatasetError, safe_json_dump, safe_json_load

logger = logging.getLogger(__name__)


@dataclass
class MMLUProQuestion:
    """Single MMLU-Pro question instance"""
    question_id: str
    subject: str
    question: str
    choices: List[str]
    correct_answer: str  # Letter (A-J)
    correct_answer_index: int
    explanation: Optional[str] = None
    difficulty: Optional[str] = None
    topic: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "question_id": self.question_id,
            "subject": self.subject,
            "question": self.question,
            "choices": self.choices,
            "correct_answer": self.correct_answer,
            "correct_answer_index": self.correct_answer_index,
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "topic": self.topic
        }


class MMLUProDataset:
    """
    Dataset class for MMLU-Pro benchmark
    Handles loading from HuggingFace or local cache
    """
    
    def __init__(self, config):
        """
        Initialize MMLU-Pro dataset
        
        Args:
            config: MMLUProConfig instance
        """
        self.config = config
        self.cache_dir = Path(config.dataset.cache_dir) if config.dataset.cache_dir else None
        self.questions_by_subject = defaultdict(list)
        self.all_questions = []
        self.subjects = []
        
        # Setup cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from HuggingFace with simplified caching"""
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / f"mmlupro_{self.config.dataset.split}_processed.json"
        
        # Try loading from cache first
        if cache_file and cache_file.exists():
            logger.info(f"Loading MMLU-Pro from cache: {cache_file}")
            try:
                cached_data = safe_json_load(str(cache_file))
                self._load_from_cached_data(cached_data)
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, reloading from HuggingFace")
        
        logger.info(f"Loading MMLU-Pro from HuggingFace: {self.config.dataset.name}")
        self._load_from_huggingface()
        
        # Save to cache
        if cache_file:
            logger.info(f"Saving to cache: {cache_file}")
            try:
                cached_data = {
                    'questions_by_subject': {k: [q.to_dict() for q in v] for k, v in self.questions_by_subject.items()},
                    'subjects': self.subjects
                }
                safe_json_dump(cached_data, str(cache_file))
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        # Filter subjects if specified
        if self.config.dataset.subjects:
            self._filter_subjects(self.config.dataset.subjects)
        
        # Limit samples per subject if specified (for testing)
        if self.config.dataset.max_samples_per_subject:
            self._limit_samples()
        
        logger.info(f"Loaded {len(self.all_questions)} questions across {len(self.subjects)} subjects")
    
    def _load_from_huggingface(self):
        """Load dataset from HuggingFace"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise DatasetError("Please install datasets: pip install datasets")
        
        try:
            # Load the dataset
            dataset = load_dataset(
                self.config.dataset.name,
                split=self.config.dataset.split,
                cache_dir=str(self.cache_dir) if self.cache_dir else None
            )
        except Exception as e:
            raise DatasetError(f"Failed to load dataset {self.config.dataset.name}: {e}")
        
        # Process each example
        for idx, example in enumerate(dataset):
            # Extract fields (adjust based on actual MMLU-Pro format)
            subject = example.get('category', 'unknown')
            
            # Create question instance
            question = MMLUProQuestion(
                question_id=f"{subject}_{idx}",
                subject=subject,
                question=example['question'],
                choices=example['options'],  # List of 10 options
                correct_answer=example['answer'],  # Should be letter A-J
                correct_answer_index=ord(example['answer']) - ord('A'),
                explanation=example.get('explanation'),
                difficulty=example.get('difficulty'),
                topic=example.get('topic')
            )
            
            self.questions_by_subject[subject].append(question)
            self.all_questions.append(question)
        
        self.subjects = sorted(list(self.questions_by_subject.keys()))
    
    def _load_from_cached_data(self, cached_data: Dict[str, Any]):
        """Load from cached JSON data"""
        # Reconstruct questions from cached data
        questions_data = cached_data['questions_by_subject']
        self.questions_by_subject = defaultdict(list)
        self.all_questions = []
        
        for subject, question_dicts in questions_data.items():
            questions = []
            for q_dict in question_dicts:
                question = MMLUProQuestion(
                    question_id=q_dict['question_id'],
                    subject=q_dict['subject'],
                    question=q_dict['question'],
                    choices=q_dict['choices'],
                    correct_answer=q_dict['correct_answer'],
                    correct_answer_index=q_dict['correct_answer_index'],
                    explanation=q_dict.get('explanation'),
                    difficulty=q_dict.get('difficulty'),
                    topic=q_dict.get('topic')
                )
                questions.append(question)
                self.all_questions.append(question)
            self.questions_by_subject[subject] = questions
        
        self.subjects = cached_data['subjects']
    
    def _filter_subjects(self, subjects: List[str]):
        """Filter dataset to specific subjects"""
        filtered_questions = defaultdict(list)
        filtered_all = []
        
        for subject in subjects:
            if subject in self.questions_by_subject:
                filtered_questions[subject] = self.questions_by_subject[subject]
                filtered_all.extend(self.questions_by_subject[subject])
        
        self.questions_by_subject = filtered_questions
        self.all_questions = filtered_all
        self.subjects = sorted(list(filtered_questions.keys()))
    
    def _limit_samples(self):
        """Limit number of samples per subject for testing"""
        limited_questions = defaultdict(list)
        limited_all = []
        
        for subject, questions in self.questions_by_subject.items():
            limited = questions[:self.config.dataset.max_samples_per_subject]
            limited_questions[subject] = limited
            limited_all.extend(limited)
        
        self.questions_by_subject = limited_questions
        self.all_questions = limited_all
    
    def get_questions_by_subject(self, subject: str) -> List[MMLUProQuestion]:
        """Get all questions for a specific subject"""
        return self.questions_by_subject.get(subject, [])
    
    def get_all_questions(self) -> List[MMLUProQuestion]:
        """Get all questions in the dataset"""
        return self.all_questions
    
    def get_few_shot_examples(self, subject: str, n: int = 5, 
                            exclude_id: Optional[str] = None) -> List[MMLUProQuestion]:
        """
        Get few-shot examples for a subject
        
        Args:
            subject: Subject to get examples from
            n: Number of examples
            exclude_id: Question ID to exclude (avoid using test question as example)
        
        Returns:
            List of example questions
        """
        questions = self.get_questions_by_subject(subject)
        examples = []
        
        for q in questions:
            if q.question_id != exclude_id:
                examples.append(q)
                if len(examples) >= n:
                    break
        
        return examples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            "total_questions": len(self.all_questions),
            "num_subjects": len(self.subjects),
            "subjects": self.subjects,
            "questions_per_subject": {
                subject: len(questions) 
                for subject, questions in self.questions_by_subject.items()
            },
            "avg_choices_per_question": 10,  # MMLU-Pro has 10 choices
        }
        return stats
    
    def __len__(self) -> int:
        """Total number of questions"""
        return len(self.all_questions)
    
    def __iter__(self):
        """Iterate over all questions"""
        return iter(self.all_questions)
    