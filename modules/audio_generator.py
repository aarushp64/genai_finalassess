"""
NeuroTutor AI - Audio Summary Generator Module
Author: Aarush Parate (Roll No: 1)
Unit IV: Generative AI Tools and Techniques
Converts AI/ML/DL explanations into high-clarity spoken audio summaries using gTTS.
"""

import io
import re
from typing import Optional, Tuple
from gtts import gTTS


class AudioSummaryGenerator:
    """Converts written technical explanations into clean spoken educational audio."""

    @staticmethod
    def clean_markdown_for_speech(text: str) -> str:
        """
        Strips markdown headers, code blocks, bold markers, and latex equations
        to produce smooth, natural spoken narration.
        """
        if not text:
            return ""

        # Remove code blocks
        clean = re.sub(r"```[\s\S]*?```", " Code snippet omitted for spoken summary. ", text)
        # Remove inline code
        clean = re.sub(r"`[^`]*`", "", clean)
        # Remove markdown links [text](url) -> text
        clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
        # Remove latex formulas $$...$$ and $...$
        clean = re.sub(r"\$\$[\s\S]*?\$\$", " mathematical formula ", clean)
        clean = re.sub(r"\$[^\$]*\$", " formula ", clean)
        # Remove headers, bullets, and bold/italic asterisks
        clean = re.sub(r"[#*_~>|]", " ", clean)
        # Remove multiple newlines and spaces
        clean = re.sub(r"\s+", " ", clean).strip()

        return clean

    @staticmethod
    def generate_speech(
        text: str,
        accent_tld: str = "com",
        slow: bool = False,
        max_chars: int = 1500,
    ) -> Tuple[Optional[bytes], str]:
        """
        Synthesizes speech from technical text.
        Returns:
            (audio_bytes, status_message)
        """
        clean_text = AudioSummaryGenerator.clean_markdown_for_speech(text)
        if not clean_text:
            return None, "No valid text provided for audio narration."

        # Truncate if explanation is very long to avoid gTTS timeout
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + "... That concludes this audio summary."

        try:
            tts = gTTS(text=clean_text, lang="en", tld=accent_tld, slow=slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            audio_bytes = buf.getvalue()
            return audio_bytes, "Audio generated successfully."
        except Exception as e:
            return None, f"Audio synthesis failed: {str(e)}"
