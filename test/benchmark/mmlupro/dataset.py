"""
Dataset module for MMLU-Pro benchmark
Handles loading, caching, and preprocessing of MMLU-Pro data
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

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
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        self.questions_by_subject = defaultdict(list)
        self.all_questions = []
        self.subjects = []
        
        # Setup cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from HuggingFace or cache"""
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / f"mmlupro_{self.config.dataset_split}.pkl"
        
        # Try loading from cache first
        if cache_file and cache_file.exists():
            logger.info(f"Loading MMLU-Pro from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.questions_by_subject = cached_data['questions_by_subject']
                self.all_questions = cached_data['all_questions']
                self.subjects = cached_data['subjects']
        else:
            logger.info(f"Loading MMLU-Pro from HuggingFace: {self.config.dataset_name}")
            self._load_from_huggingface()
            
            # Save to cache
            if cache_file:
                logger.info(f"Saving to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'questions_by_subject': self.questions_by_subject,
                        'all_questions': self.all_questions,
                        'subjects': self.subjects
                    }, f)
        
        # Filter subjects if specified
        if self.config.subjects:
            self._filter_subjects(self.config.subjects)
        
        # Limit samples per subject if specified (for testing)
        if self.config.max_samples_per_subject:
            self._limit_samples()
        
        logger.info(f"Loaded {len(self.all_questions)} questions across {len(self.subjects)} subjects")
    
    def _load_from_huggingface(self):
        """Load dataset from HuggingFace"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        # Load the dataset
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None
        )
        
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
            limited = questions[:self.config.max_samples_per_subject]
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
    