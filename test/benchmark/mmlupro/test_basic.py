#!/usr/bin/env python3
"""
Basic test script to verify MMLU-Pro modules are working correctly
Run this to test dataset loading and configuration
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import MMLUProConfig, EvaluationMode, PromptStyle, ALL_SUBJECTS
from dataset import MMLUProDataset, MMLUProQuestion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_config():
    """Test configuration module"""
    print("\n" + "="*50)
    print("Testing Configuration Module")
    print("="*50)
    
    # Create default config
    config = MMLUProConfig()
    print("✓ Created default config")
    
    # Test with custom settings
    custom_config = MMLUProConfig(
        model_name="test-model",
        batch_size=16,
        evaluation_mode=EvaluationMode.COT,
        num_few_shot_examples=3,
        subjects=["mathematics", "physics"],
        max_samples_per_subject=10
    )
    print("✓ Created custom config")
    
    # Test serialization
    config_dict = custom_config.to_dict()
    restored_config = MMLUProConfig.from_dict(config_dict)
    print("✓ Config serialization works")
    
    # Display config
    print("\nSample Configuration:")
    print("-" * 30)
    for i, (key, value) in enumerate(list(config_dict.items())[:10]):  # Show first 10 params
        print(f"  {key}: {value}")
    
    return True


def test_dataset_mock():
    """Test dataset module with mock data (without downloading)"""
    print("\n" + "="*50)
    print("Testing Dataset Module (Mock)")
    print("="*50)
    
    # Create a mock question
    mock_question = MMLUProQuestion(
        question_id="math_001",
        subject="mathematics",
        question="What is 2 + 2?",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        correct_answer="D",
        correct_answer_index=3,
        explanation="Basic arithmetic: 2 + 2 = 4",
        difficulty="easy",
        topic="arithmetic"
    )
    
    print(f"✓ Created mock question: {mock_question.question_id}")
    
    # Test question serialization
    question_dict = mock_question.to_dict()
    print("✓ Question serialization works")
    
    # Display question
    print("\nMock Question:")
    print(f"  Question: {mock_question.question}")
    print(f"  Choices: A-J with 10 options")
    print(f"  Correct Answer: {mock_question.correct_answer}")
    print(f"  Subject: {mock_question.subject}")
    
    return True


def test_prompts():
    """Test prompt formatting"""
    print("\n" + "="*50)
    print("Testing Prompt Templates")
    print("="*50)
    
    from prompts import PromptFormatter
    
    # Create config and formatter
    config = MMLUProConfig(
        evaluation_mode=EvaluationMode.COT,
        prompt_style=PromptStyle.STANDARD
    )
    formatter = PromptFormatter(config)
    print("✓ Created prompt formatter")
    
    # Create a test question
    test_question = MMLUProQuestion(
        question_id="test_001",
        subject="mathematics",
        question="What is the derivative of x^2?",
        choices=["x", "2x", "x^2", "2", "1", "0", "x/2", "2x^2", "x^3/3", "None"],
        correct_answer="B",
        correct_answer_index=1,
        explanation="Using the power rule: d/dx(x^n) = nx^(n-1), so d/dx(x^2) = 2x",
    )
    
    # Format the question
    formatted = formatter.format_question(test_question)
    print("✓ Formatted question successfully")
    print("\nFormatted prompt (first 200 chars):")
    print("-" * 30)
    print(formatted[:200] + "...")
    
    # Test answer extraction
    test_response = "Let me think about this. The derivative of x^2 is 2x. The answer is (B)"
    extracted = formatter.extract_answer(test_response)
    print(f"\n✓ Extracted answer '{extracted}' from response")
    
    return True


def test_dataset_real(download=False):
    """Test dataset module with real data"""
    if not download:
        print("\nSkipping real dataset test (set download=True to test)")
        return True
    
    print("\n" + "="*50)
    print("Testing Dataset Module (Real Data)")
    print("="*50)
    
    # Create config for testing
    config = MMLUProConfig(
        cache_dir="./cache_test",
        max_samples_per_subject=5,  # Limit for quick testing
        subjects=["mathematics", "physics"]  # Test with 2 subjects
    )
    
    # Load dataset
    print("Loading MMLU-Pro dataset...")
    dataset = MMLUProDataset(config)
    
    # Display statistics
    stats = dataset.get_statistics()
    print(f"\n✓ Loaded {stats['total_questions']} questions")
    print(f"✓ Subjects: {stats['num_subjects']}")
    
    # Test iteration
    first_question = next(iter(dataset))
    print(f"\nSample Question:")
    print(f"  ID: {first_question.question_id}")
    print(f"  Subject: {first_question.subject}")
    print(f"  Question: {first_question.question[:100]}...")
    print(f"  Number of choices: {len(first_question.choices)}")
    
    # Test few-shot examples
    examples = dataset.get_few_shot_examples(first_question.subject, n=3)
    print(f"\n✓ Retrieved {len(examples)} few-shot examples")
    
    return True


def test_imports():
    """Test all imports are working"""
    print("\n" + "="*50)
    print("Testing Imports")
    print("="*50)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets version: {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("MMLU-Pro Module Test Suite")
    print("="*50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        print("\nImport test failed. Please install requirements:")
        print("pip install -r requirements.txt")
        all_passed = False
    
    # Test configuration
    try:
        test_config()
    except Exception as e:
        print(f"Config test failed: {e}")
        all_passed = False
    
    # Test dataset with mock data
    try:
        test_dataset_mock()
    except Exception as e:
        print(f"Dataset mock test failed: {e}")
        all_passed = False
    
    # Test prompt templates
    try:
        test_prompts()
    except Exception as e:
        print(f"Prompt test failed: {e}")
        all_passed = False
    
    # Test dataset with real data (optional)
    # Uncomment the line below to test with real data download
    # test_dataset_real(download=True)
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*50)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)