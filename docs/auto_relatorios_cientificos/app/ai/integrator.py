"""
AI integration module for AUTO-RELATÓRIOS CIENTÍFICOS
Sprint 3 - AI-powered text enhancement
"""

import os
from typing import Dict, Any, Optional
from ..config import CONFIG

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class AILanguageEnhancer:
    """AI-powered language enhancement for scientific texts"""

    def __init__(self):
        """Initialize AI enhancer with configuration"""
        self.config = CONFIG.get("ai", {})
        self.enable_ai_layer = self.config.get("enable_ai_layer", False)
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-4-turbo")
        self.temperature = self.config.get("temperature", 0.3)

        # Initialize clients if enabled
        self.openai_client = None
        self.anthropic_client = None

        if self.enable_ai_layer and OPENAI_AVAILABLE:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)

        if self.enable_ai_layer and ANTHROPIC_AVAILABLE:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                self.anthropic_client = Anthropic(api_key=anthropic_api_key)

    def enhance_text(self, text: str, context: str = "", enhancement_type: str = "scientific") -> str:
        """
        Enhance text using AI

        Args:
            text (str): Text to enhance
            context (str): Context for enhancement
            enhancement_type (str): Type of enhancement (scientific, formal, concise, etc.)

        Returns:
            str: Enhanced text
        """
        if not self.enable_ai_layer or not text.strip():
            return text

        try:
            if self.provider == "openai" and self.openai_client:
                return self._enhance_with_openai(text, context, enhancement_type)
            elif self.provider == "anthropic" and self.anthropic_client:
                return self._enhance_with_anthropic(text, context, enhancement_type)
            else:
                # Fallback to original text if no provider available
                return text
        except Exception as e:
            # Return original text if enhancement fails
            return text

    def _enhance_with_openai(self, text: str, context: str, enhancement_type: str) -> str:
        """Enhance text using OpenAI"""
        prompt = self._create_enhancement_prompt(text, context, enhancement_type)

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a scientific writing assistant that improves text clarity, grammar, and professionalism while maintaining accuracy."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=4096
        )

        return response.choices[0].message.content.strip()

    def _enhance_with_anthropic(self, text: str, context: str, enhancement_type: str) -> str:
        """Enhance text using Anthropic"""
        prompt = self._create_enhancement_prompt(text, context, enhancement_type)

        response = self.anthropic_client.messages.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are a scientific writing assistant that improves text clarity, grammar, and professionalism while maintaining accuracy.",
            temperature=self.temperature,
            max_tokens=4096
        )

        return response.content[0].text.strip()

    def _create_enhancement_prompt(self, text: str, context: str, enhancement_type: str) -> str:
        """Create enhancement prompt based on type"""
        base_prompt = f"Please enhance the following scientific text:\n\n{text}\n\n"

        if enhancement_type == "scientific":
            enhancement_instruction = (
                "Improve this scientific text by:\n"
                "1. Enhancing clarity and readability\n"
                "2. Improving grammar and sentence structure\n"
                "3. Maintaining technical accuracy\n"
                "4. Using appropriate scientific terminology\n"
                "5. Ensuring professional tone\n\n"
                "Return only the enhanced text without additional explanations."
            )
        elif enhancement_type == "formal":
            enhancement_instruction = (
                "Convert this text to formal scientific language by:\n"
                "1. Using passive voice where appropriate\n"
                "2. Employing precise technical terminology\n"
                "3. Maintaining objective tone\n"
                "4. Following academic writing conventions\n\n"
                "Return only the enhanced text without additional explanations."
            )
        elif enhancement_type == "concise":
            enhancement_instruction = (
                "Make this text more concise by:\n"
                "1. Removing redundant words and phrases\n"
                "2. Combining related sentences\n"
                "3. Using more direct language\n"
                "4. Maintaining key information\n\n"
                "Return only the enhanced text without additional explanations."
            )
        else:
            enhancement_instruction = (
                "Improve this text by enhancing clarity, grammar, and professionalism "
                "while maintaining the original meaning and technical accuracy.\n\n"
                "Return only the enhanced text without additional explanations."
            )

        return base_prompt + enhancement_instruction

# Global instance
ai_enhancer = AILanguageEnhancer()

def enhance_section_text(text: str, context: str = "", enhancement_type: str = "scientific") -> str:
    """
    Enhance section text using AI

    Args:
        text (str): Text to enhance
        context (str): Context for enhancement
        enhancement_type (str): Type of enhancement

    Returns:
        str: Enhanced text
    """
    return ai_enhancer.enhance_text(text, context, enhancement_type)