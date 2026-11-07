"""
NeuroTutor AI - Safety Guardrails & Input Validation Module
Author: Aarush Parate (Roll No: 1)
Unit V: Responsible AI and Ethics
"""

import re
from typing import Tuple, Dict, Any, List

# Signatures of prompt injection and system jailbreak attempts
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"bypass\s+all\s+(rules|guidelines|safety|protocols)",
    r"system\s*:\s*override",
    r"jailbreak",
    r"do\s+anything\s+now\s+(dan)",
    r"forget\s+your\s+training",
    r"pretend\s+you\s+have\s+no\s+(morals|filters|restrictions)",
    r"roleplay\s+as\s+an\s+evil",
    r"give\s+me\s+your\s+system\s+prompt",
    r"repeat\s+the\s+words\s+above",
    r"<\|\s*im_start\s*\|>",
    r"<\|\s*system\s*\|>",
]

# Sensitive/Harmful content keywords outside educational scope
HARMFUL_PATTERNS = [
    r"\b(make|synthesize|create)\s+(a\s+)?(bomb|explosive|weapon|poison)\b",
    r"\bhow\s+to\s+hack\s+(into\s+)?(a\s+)?(bank|server|government|network)\b",
    r"\bmalware\s+(script|payload|injection|code)\b",
    r"\bcredit\s*card\s*(number|fraud|stealing)\b",
    r"\bddos\s+attack\s+script\b",
]

# Maximum permitted prompt length in characters
MAX_PROMPT_LENGTH = 4000
MIN_PROMPT_LENGTH = 2


class SafetyGuardrails:
    """Provides input validation, prompt injection defense, and harmful content filtering."""

    @staticmethod
    def check_prompt_injection(text: str) -> Tuple[bool, str]:
        """
        Scans input for prompt injection signatures and unauthorized instruction overrides.
        Returns:
            (is_safe: bool, reason: str)
        """
        text_lower = text.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return False, f"Prompt injection signature detected: '{pattern}'"
        return True, "No injection signatures detected."

    @staticmethod
    def check_harmful_content(text: str) -> Tuple[bool, str]:
        """
        Detects requests that fall outside ethical educational guidelines.
        Returns:
            (is_safe: bool, reason: str)
        """
        text_lower = text.lower()
        for pattern in HARMFUL_PATTERNS:
            if re.search(pattern, text_lower):
                return False, "Harmful or non-educational content detected."
        return True, "Content passed ethical safety checks."

    @staticmethod
    def validate_user_prompt(text: str) -> Tuple[bool, str]:
        """
        Comprehensive prompt validation pipeline.
        Returns:
            (is_valid: bool, feedback_message: str)
        """
        if not text or not text.strip():
            return False, "Prompt cannot be empty."

        clean_text = text.strip()
        if len(clean_text) < MIN_PROMPT_LENGTH:
            return False, f"Prompt too short. Minimum length is {MIN_PROMPT_LENGTH} characters."

        if len(clean_text) > MAX_PROMPT_LENGTH:
            return False, f"Prompt exceeds maximum character limit of {MAX_PROMPT_LENGTH} characters."

        # Check injection
        is_safe, msg = SafetyGuardrails.check_prompt_injection(clean_text)
        if not is_safe:
            return False, f"Security Warning: {msg}"

        # Check harmful content
        is_safe, msg = SafetyGuardrails.check_harmful_content(clean_text)
        if not is_safe:
            return False, f"Ethics Warning: {msg}"

        return True, "Prompt is safe and valid."

    @staticmethod
    def sanitize_document_text(text: str) -> str:
        """
        Sanitizes untrusted text extracted from student-uploaded documents.
        Prevents prompt injection within RAG context by wrapping and filtering delimiter attacks.
        """
        if not text:
            return ""

        # Remove control characters except newline and tab
        sanitized = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

        # Defuse potential system tags that an attacker might embed in a PDF
        sanitized = re.sub(r"<\|im_start\|>", "[TAG_FILTERED]", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"<\|system\|>", "[TAG_FILTERED]", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"---BEGIN SYSTEM INSTRUCTION---", "[TAG_FILTERED]", sanitized, flags=re.IGNORECASE)

        return sanitized.strip()

    @staticmethod
    def validate_file_upload(file_name: str, file_size_bytes: int) -> Tuple[bool, str]:
        """Validates uploaded files by extension and size."""
        allowed_extensions = {".pdf", ".txt", ".docx"}
        max_size_mb = 10
        max_size_bytes = max_size_mb * 1024 * 1024

        extension = "." + file_name.split(".")[-1].lower() if "." in file_name else ""
        if extension not in allowed_extensions:
            return False, f"Unsupported file type '{extension}'. Allowed formats: PDF, TXT, DOCX."

        if file_size_bytes > max_size_bytes:
            return False, f"File size exceeds {max_size_mb} MB limit ({file_size_bytes / (1024*1024):.2f} MB)."

        return True, "File verified successfully."
