"""
NeuroTutor AI - Conversation Memory Module
Author: Aarush Parate (Roll No: 1)
Unit II: GANs and Large Language Models
Provides bounded conversation memory, context compilation, and state tracking.
"""

from typing import List, Dict, Any, Optional
import time
import config


class ConversationMemoryManager:
    """Manages conversational session history with token-bounded context windowing."""

    def __init__(self, max_turns: int = config.MAX_HISTORY_TURNS):
        self.max_turns = max_turns

    @staticmethod
    def get_or_create_history(session_state) -> List[Dict[str, Any]]:
        """Ensures session history exists in Streamlit state."""
        if "conversation_history" not in session_state:
            session_state.conversation_history = []
        return session_state.conversation_history

    def add_turn(
        self,
        session_state,
        user_message: str,
        assistant_message: str,
        level: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Appends a completed dialogue turn and enforces window bounds."""
        history = self.get_or_create_history(session_state)
        turn = {
            "timestamp": time.time(),
            "level": level,
            "user": user_message.strip(),
            "assistant": assistant_message.strip(),
            "metadata": metadata or {},
        }
        history.append(turn)

        # Truncate older history to prevent unbounded context growth
        if len(history) > self.max_turns:
            session_state.conversation_history = history[-self.max_turns :]

    @staticmethod
    def clear_history(session_state) -> None:
        """Resets conversational memory."""
        session_state.conversation_history = []

    def compile_context_prompt(
        self, session_state, current_question: str, current_level: str
    ) -> str:
        """
        Formats recent turns into a structured context window for the LLM.
        """
        history = self.get_or_create_history(session_state)

        if not history:
            return (
                f"You are NeuroTutor AI, an expert AI tutor.\n"
                f"Audience Knowledge Level: {current_level}.\n\n"
                f"Student Question: {current_question}\n\n"
                f"Detailed Explanation:"
            )

        context_lines = [
            f"You are NeuroTutor AI. You are having an ongoing educational dialogue.",
            f"Audience Knowledge Level: {current_level}.",
            f"Here is the recent conversation history:",
            "--- Recent Dialogue Context ---",
        ]

        for idx, turn in enumerate(history[-self.max_turns :]):
            context_lines.append(f"Student: {turn['user']}")
            context_lines.append(f"Tutor ({turn['level']}): {turn['assistant']}")
            context_lines.append("")

        context_lines.append("--- End of Context ---")
        context_lines.append(
            f"Now address the student's follow-up question accurately at the '{current_level}' level, maintaining continuity with the previous explanation."
        )
        context_lines.append(f"Student Follow-up: {current_question}")
        context_lines.append("Tutor Response:")

        return "\n".join(context_lines)
