"""
Prompt templates and formatting for MMLU-Pro evaluation
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import re
from abc import ABC, abstractmethod

from dataset import MMLUProQuestion
from config import PromptStyle, EvaluationMode


class BasePromptTemplate(ABC):
    """Base class for prompt templates"""
    
    @abstractmethod
    def format_question(self, question: MMLUProQuestion, 
                       examples: Optional[List[MMLUProQuestion]] = None) -> str:
        """Format a question into a prompt"""
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from model response"""
        pass


class StandardPromptTemplate(BasePromptTemplate):
    """Standard QA format prompt template"""
    
    def __init__(self, config):
        self.config = config
        self.cot_trigger = config.cot_trigger
        self.answer_pattern = re.compile(config.extract_answer_pattern, re.IGNORECASE)
    
    def format_question(self, question: MMLUProQuestion, 
                       examples: Optional[List[MMLUProQuestion]] = None) -> str:
        """Format question in standard QA format"""
        prompt_parts = []
        
        # Add few-shot examples if provided
        if examples:
            prompt_parts.append("Here are some example questions:\n")
            for ex in examples:
                prompt_parts.append(self._format_single_example(ex, include_answer=True))
                prompt_parts.append("")  # Empty line between examples
        
        # Add the actual question
        if examples:
            prompt_parts.append("Now answer this question:\n")
        
        prompt_parts.append(self._format_single_example(question, include_answer=False))
        
        # Add CoT trigger if needed
        if self.config.evaluation_mode == EvaluationMode.COT:
            prompt_parts.append(f"\n{self.cot_trigger}")
            prompt_parts.append("\nProvide your reasoning step by step, then give your final answer in the format: 'The answer is (X)' where X is the letter of the correct option.")
        else:
            prompt_parts.append("\nAnswer with only the letter of the correct option.")
        
        return "\n".join(prompt_parts)
    
    def _format_single_example(self, question: MMLUProQuestion, include_answer: bool) -> str:
        """Format a single question"""
        lines = []
        lines.append(f"Question: {question.question}")
        lines.append("\nOptions:")
        
        # Format choices with letters A-J
        for idx, choice in enumerate(question.choices):
            letter = chr(ord('A') + idx)
            lines.append(f"{letter}. {choice}")
        
        if include_answer:
            if self.config.evaluation_mode == EvaluationMode.COT and question.explanation:
                lines.append(f"\nReasoning: {question.explanation}")
            lines.append(f"\nAnswer: The answer is ({question.correct_answer})")
        
        return "\n".join(lines)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer letter from response"""
        # Try to find pattern like "The answer is (X)" or "The answer is X"
        match = self.answer_pattern.search(response)
        if match:
            answer = match.group(1).upper()
            if answer in 'ABCDEFGHIJ':
                return answer
        
        # Fallback: Look for standalone letter at the end
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # Check if line is just a single letter
            if len(line) == 1 and line in 'ABCDEFGHIJ':
                return line
            # Check if line starts with a letter followed by period or parenthesis
            if len(line) >= 2 and line[0] in 'ABCDEFGHIJ' and line[1] in '.):':
                return line[0]
        
        return None


class InstructionPromptTemplate(BasePromptTemplate):
    """Instruction-following format prompt template"""
    
    def __init__(self, config):
        self.config = config
        self.answer_pattern = re.compile(config.extract_answer_pattern, re.IGNORECASE)
    
    def format_question(self, question: MMLUProQuestion, 
                       examples: Optional[List[MMLUProQuestion]] = None) -> str:
        """Format question as instruction"""
        instruction = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        instruction += f"You are solving multiple-choice questions in {question.subject}. "
        
        if self.config.evaluation_mode == EvaluationMode.COT:
            instruction += "Think through the problem step by step, then provide your final answer."
        else:
            instruction += "Select the best answer from the given options."
        
        prompt_parts = [instruction, ""]
        
        # Add few-shot examples
        if examples:
            for i, ex in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(self._format_example(ex, include_answer=True))
                prompt_parts.append("")
        
        
        prompt_parts.append("<|eot_id|><|start_header_id|>user<|end_header_id|>")

        # Add the actual question
        prompt_parts.append("Question to solve:")
        prompt_parts.append(self._format_example(question, include_answer=False))
        
        if self.config.evaluation_mode == EvaluationMode.COT:
            prompt_parts.append("\nSolution:")
        else:
            prompt_parts.append("\nAnswer:")
        
        prompt_parts.append("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
        
        return "\n".join(prompt_parts)
    
    def _format_example(self, question: MMLUProQuestion, include_answer: bool) -> str:
        """Format a single example"""
        lines = [question.question, "\nChoices:"]
        
        for idx, choice in enumerate(question.choices):
            letter = chr(ord('A') + idx)
            lines.append(f"  {letter}) {choice}")
        
        if include_answer:
            if self.config.evaluation_mode == EvaluationMode.COT and question.explanation:
                lines.append(f"\nSolution: {question.explanation}")
            lines.append(f"Final Answer: {question.correct_answer}")
        
        return "\n".join(lines)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from instruction response"""
        # Similar to standard template
        match = self.answer_pattern.search(response)
        if match:
            answer = match.group(1).upper()
            if answer in 'ABCDEFGHIJ':
                return answer
        
        # Look for "Final Answer: X" pattern
        final_answer_pattern = r"[Ff]inal [Aa]nswer:?\s*\(?([A-J])\)?"
        match = re.search(final_answer_pattern, response)
        if match:
            return match.group(1).upper()
        
        # Fallback to looking for isolated letter
        for line in response.strip().split('\n'):
            line = line.strip()
            if len(line) == 1 and line.upper() in 'ABCDEFGHIJ':
                return line.upper()
        
        return None


class ChatPromptTemplate(BasePromptTemplate):
    """Chat format with system message"""
    
    def __init__(self, config):
        self.config = config
        self.system_prompt = config.system_prompt or self._default_system_prompt()
        self.answer_pattern = re.compile(config.extract_answer_pattern, re.IGNORECASE)
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for chat format"""
        if self.config.evaluation_mode == EvaluationMode.COT:
            return (
                "You are an expert problem solver. When presented with multiple-choice questions, "
                "think through the problem step by step, showing your reasoning clearly. "
                "Always end your response with 'The answer is (X)' where X is the letter of your chosen option."
            )
        else:
            return (
                "You are an expert problem solver. When presented with multiple-choice questions, "
                "select the best answer from the given options. "
                "Respond with only the letter of the correct option."
            )
    
    def format_question(self, question: MMLUProQuestion, 
                       examples: Optional[List[MMLUProQuestion]] = None) -> Dict[str, str]:
        """Format as chat messages"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add few-shot examples as conversation history
        if examples:
            for ex in examples:
                # User asks the question
                user_msg = self._format_question_only(ex)
                messages.append({"role": "user", "content": user_msg})
                
                # Assistant provides answer
                if self.config.evaluation_mode == EvaluationMode.COT and ex.explanation:
                    assistant_msg = f"{ex.explanation}\n\nThe answer is ({ex.correct_answer})"
                else:
                    assistant_msg = f"The answer is ({ex.correct_answer})"
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add the actual question
        user_msg = self._format_question_only(question)
        messages.append({"role": "user", "content": user_msg})
        
        return messages
    
    def _format_question_only(self, question: MMLUProQuestion) -> str:
        """Format just the question and choices"""
        lines = [f"Subject: {question.subject}", f"\n{question.question}", "\nOptions:"]
        
        for idx, choice in enumerate(question.choices):
            letter = chr(ord('A') + idx)
            lines.append(f"{letter}. {choice}")
        
        return "\n".join(lines)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from chat response"""
        # Same extraction logic as other templates
        match = self.answer_pattern.search(response)
        if match:
            answer = match.group(1).upper()
            if answer in 'ABCDEFGHIJ':
                return answer
        
        # Fallback
        for line in response.strip().split('\n'):
            line = line.strip()
            if len(line) == 1 and line.upper() in 'ABCDEFGHIJ':
                return line.upper()
        
        return None


class PromptFormatter:
    """Main prompt formatter that selects appropriate template"""
    
    def __init__(self, config):
        self.config = config
        self.template = self._get_template()
    
    def _get_template(self) -> BasePromptTemplate:
        """Get appropriate template based on config"""
        if self.config.prompt_style == PromptStyle.STANDARD:
            return StandardPromptTemplate(self.config)
        elif self.config.prompt_style == PromptStyle.INSTRUCTION:
            return InstructionPromptTemplate(self.config)
        elif self.config.prompt_style == PromptStyle.CHAT:
            return ChatPromptTemplate(self.config)
        else:
            raise ValueError(f"Unknown prompt style: {self.config.prompt_style}")
    
    def format_question(self, question: MMLUProQuestion, 
                       examples: Optional[List[MMLUProQuestion]] = None) -> Any:
        """Format question using selected template"""
        return self.template.format_question(question, examples)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer using selected template"""
        return self.template.extract_answer(response)
    
    def format_batch(self, questions: List[MMLUProQuestion], 
                    examples_dict: Optional[Dict[str, List[MMLUProQuestion]]] = None) -> List[Any]:
        """Format a batch of questions"""
        formatted = []
        for q in questions:
            examples = examples_dict.get(q.question_id) if examples_dict else None
            formatted.append(self.format_question(q, examples))
        return formatted