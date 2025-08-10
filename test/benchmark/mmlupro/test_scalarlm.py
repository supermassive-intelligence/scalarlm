#!/usr/bin/env python3
"""
Test ScalarLM integration for MMLU-Pro benchmark
This script directly tests the ScalarLM generate() calls
"""

import logging
import scalarlm
from config import MMLUProConfig, EvaluationMode, PromptStyle
from dataset import MMLUProQuestion
from prompts import PromptFormatter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set ScalarLM API URL
scalarlm.api_url = "http://localhost:8000"

print(f"ScalarLM API URL set to: {scalarlm.api_url}")


def test_direct_scalarlm():
    """Test direct ScalarLM call"""
    print("\n" + "="*50)
    print("Testing Direct ScalarLM Integration")
    print("="*50)
    
    # Initialize ScalarLM
    llm = scalarlm.SupermassiveIntelligence()
    print("✓ Initialized ScalarLM")
    
    # Test health check
    try:
        health = llm.health()
        print(f"✓ Health check: {health}")
    except Exception as e:
        print(f"⚠ Health check failed: {e}")
    
    # Test simple generation
    test_prompt = "What is 2 + 2? Answer with just the number."
    print(f"\nTest prompt: {test_prompt}")
    
    try:
        responses = llm.generate(prompts=[test_prompt])
        print(f"✓ Generate response: {responses[0] if responses else 'No response'}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False
    
    return True


def test_mmlupro_formatted_prompt():
    """Test ScalarLM with MMLU-Pro formatted prompt"""
    print("\n" + "="*50)
    print("Testing MMLU-Pro Formatted Prompt with ScalarLM")
    print("="*50)
    
    # Create a test question
    test_question = MMLUProQuestion(
        question_id="test_001",
        subject="mathematics",
        question="What is the derivative of x^2 with respect to x?",
        choices=["x", "2x", "x^2", "2", "1", "0", "x/2", "2x^2", "x^3/3", "None of these"],
        correct_answer="B",
        correct_answer_index=1,
        explanation="Using the power rule: d/dx(x^n) = nx^(n-1), so d/dx(x^2) = 2x"
    )
    print(f"✓ Created test question")
    
    # Create formatter
    config = MMLUProConfig(
        evaluation_mode=EvaluationMode.COT,
        prompt_style=PromptStyle.STANDARD
    )
    formatter = PromptFormatter(config)
    print(f"✓ Created prompt formatter")
    
    # Format the prompt
    formatted_prompt = formatter.format_question(test_question)
    print(f"\nFormatted prompt (first 300 chars):")
    print("-" * 30)
    print(formatted_prompt[:300] + "...")
    
    # Call ScalarLM
    llm = scalarlm.SupermassiveIntelligence()
    
    try:
        print("\nCalling ScalarLM generate()...")
        responses = llm.generate(prompts=[formatted_prompt])
        response = responses[0] if responses else ""
        
        print(f"\n✓ Got response ({len(response)} chars)")
        print("Response (first 500 chars):")
        print("-" * 30)
        print(response[:500])
        
        # Extract answer
        extracted_answer = formatter.extract_answer(response)
        print(f"\n✓ Extracted answer: {extracted_answer}")
        print(f"✓ Correct answer: {test_question.correct_answer}")
        print(f"✓ Match: {extracted_answer == test_question.correct_answer}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False
    
    return True


def test_batch_generation():
    """Test batch generation with ScalarLM"""
    print("\n" + "="*50)
    print("Testing Batch Generation with ScalarLM")
    print("="*50)
    
    # Create multiple test prompts
    test_prompts = [
        "What is 2 + 2? Answer: The answer is",
        "What is 5 * 3? Answer: The answer is",
        "What is 10 - 7? Answer: The answer is"
    ]
    
    print(f"Created {len(test_prompts)} test prompts")
    
    # Call ScalarLM with batch
    llm = scalarlm.SupermassiveIntelligence()
    
    try:
        print("\nCalling ScalarLM with batch...")
        responses = llm.generate(prompts=test_prompts)
        
        print(f"✓ Got {len(responses)} responses")
        for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Response: {response}")
        
    except Exception as e:
        print(f"✗ Batch generation failed: {e}")
        return False
    
    return True


def test_with_model_name():
    """Test generation with specific model name"""
    print("\n" + "="*50)
    print("Testing Generation with Model Name")
    print("="*50)
    
    model_name = "your-model-name"  # Replace with actual model name
    print(f"Testing with model: {model_name}")
    
    llm = scalarlm.SupermassiveIntelligence()
    
    try:
        responses = llm.generate(
            prompts=["What is 2 + 2?"],
            model_name=model_name
        )
        print(f"✓ Generated with model {model_name}: {responses[0] if responses else 'No response'}")
    except Exception as e:
        print(f"⚠ Generation with model name failed: {e}")
        print("  This might be expected if the model name doesn't exist")
    
    return True


def main():
    """Run all ScalarLM integration tests"""
    print("\n" + "="*50)
    print("ScalarLM Integration Test Suite")
    print("="*50)
    
    all_passed = True
    
    # Test 1: Direct ScalarLM
    try:
        if not test_direct_scalarlm():
            all_passed = False
    except Exception as e:
        print(f"Direct test failed: {e}")
        all_passed = False
    
    # Test 2: MMLU-Pro formatted prompt
    try:
        if not test_mmlupro_formatted_prompt():
            all_passed = False
    except Exception as e:
        print(f"MMLU-Pro prompt test failed: {e}")
        all_passed = False
    
    # Test 3: Batch generation
    try:
        if not test_batch_generation():
            all_passed = False
    except Exception as e:
        print(f"Batch test failed: {e}")
        all_passed = False
    
    # Test 4: With model name
    try:
        test_with_model_name()  # This one is allowed to fail
    except Exception as e:
        print(f"Model name test failed: {e}")
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ All core tests passed!")
        print("\nYou can now run the full benchmark with:")
        print("  python main.py --debug")
    else:
        print("✗ Some tests failed")
        print("Please check your ScalarLM setup")
    print("="*50)


if __name__ == "__main__":
    main()