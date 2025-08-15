"""
Prompt templates and formatting for MMLU-Pro evaluation (Refactored)
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dataset import MMLUProQuestion
from config import PromptStyle, EvaluationMode
from utils import AnswerExtractor, format_choices_with_letters


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
        self.cot_trigger = config.evaluation.cot_trigger
        self.answer_extractor = AnswerExtractor(config.evaluation.extract_answer_pattern)
    
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
        if self.config.evaluation.evaluation_mode == EvaluationMode.COT:
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
        formatted_choices = format_choices_with_letters(question.choices)
        lines.extend(formatted_choices)
        
        if include_answer:
            if self.config.evaluation.evaluation_mode == EvaluationMode.COT and question.explanation:
                lines.append(f"\nReasoning: {question.explanation}")
            lines.append(f"\nAnswer: The answer is ({question.correct_answer})")
        
        return "\n".join(lines)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer letter from response"""
        return self.answer_extractor.extract_letter(response)


class InstructionPromptTemplate(BasePromptTemplate):
    """Instruction-following format prompt template"""
    
    def __init__(self, config):
        self.config = config
        self.answer_extractor = AnswerExtractor(config.evaluation.extract_answer_pattern)
    
    def format_question(self, question: MMLUProQuestion, 
                       examples: Optional[List[MMLUProQuestion]] = None) -> str:
        """Format question as instruction"""
        instruction = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        instruction += f"You are solving multiple-choice questions in {question.subject}. "
        
        if self.config.evaluation.evaluation_mode == EvaluationMode.COT:
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
        
        if self.config.evaluation.evaluation_mode == EvaluationMode.COT:
            prompt_parts.append("\nSolution:")
        else:
            prompt_parts.append("\nAnswer:")
        
        prompt_parts.append("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
        
        return "\n".join(prompt_parts)
    
    def _format_example(self, question: MMLUProQuestion, include_answer: bool) -> str:
        """Format a single example"""
        lines = [question.question, "\nChoices:"]
        
        formatted_choices = format_choices_with_letters(question.choices)
        for choice in formatted_choices:
            lines.append(f"  {choice.replace('.', ')')}")
        
        if include_answer:
            if self.config.evaluation.evaluation_mode == EvaluationMode.COT and question.explanation:
                lines.append(f"\nSolution: {question.explanation}")
            lines.append(f"Final Answer: {question.correct_answer}")
        
        return "\n".join(lines)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from instruction response"""
        # Try additional patterns specific to instruction format
        additional_patterns = [r"[Ff]inal [Aa]nswer:?\s*\(?([A-J])\)?"]
        return self.answer_extractor.extract_letter(response, additional_patterns)


class ChatPromptTemplate(BasePromptTemplate):
    """Chat format with system message"""
    
    def __init__(self, config):
        self.config = config
        self.system_prompt = config.advanced.system_prompt or self._default_system_prompt()
        self.answer_extractor = AnswerExtractor(config.evaluation.extract_answer_pattern)
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for chat format"""
        if self.config.evaluation.evaluation_mode == EvaluationMode.COT:
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
                if self.config.evaluation.evaluation_mode == EvaluationMode.COT and ex.explanation:
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
        
        formatted_choices = format_choices_with_letters(question.choices)
        lines.extend(formatted_choices)
        
        return "\n".join(lines)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from chat response"""
        return self.answer_extractor.extract_letter(response)


# Template registry for simple selection
TEMPLATES = {
    PromptStyle.STANDARD: StandardPromptTemplate,
    PromptStyle.INSTRUCTION: InstructionPromptTemplate,
    PromptStyle.CHAT: ChatPromptTemplate
}


class PromptFormatter:
    """Main prompt formatter that selects appropriate template"""
    
    def __init__(self, config):
        self.config = config
        self.template = self._get_template()
    
    def _get_template(self) -> BasePromptTemplate:
        """Get appropriate template based on config"""
        template_class = TEMPLATES.get(self.config.evaluation.prompt_style)
        if template_class is None:
            raise ValueError(f"Unknown prompt style: {self.config.evaluation.prompt_style}")
        return template_class(self.config)
    
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