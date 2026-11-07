"""
NeuroTutor AI - Document Processor Module
Author: Aarush Parate (Roll No: 1)
Unit III: GPT, BERT and Multimodal Models
Extracts text from PDF, DOCX, and TXT files, cleans content, and generates overlapping chunks.
"""

import io
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
import docx
from modules.safety_guardrails import SafetyGuardrails
import config


class DocumentProcessor:
    """Handles multi-format academic note text extraction, sanitization, and chunking."""

    @staticmethod
    def extract_text_from_bytes(file_bytes: bytes, file_name: str) -> str:
        """
        Extracts clean text based on file extension (PDF, TXT, DOCX).
        """
        extension = "." + file_name.split(".")[-1].lower() if "." in file_name else ""
        extracted_text = ""

        if extension == ".pdf":
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                pages = []
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(f"[Page {idx + 1}]\n{page_text}")
                extracted_text = "\n\n".join(pages)
            except Exception as e:
                raise ValueError(f"Error reading PDF file: {str(e)}")

        elif extension == ".docx":
            try:
                doc = docx.Document(io.BytesIO(file_bytes))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                extracted_text = "\n".join(paragraphs)
            except Exception as e:
                raise ValueError(f"Error reading DOCX file: {str(e)}")

        elif extension == ".txt":
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("latin-1", errors="ignore")

        else:
            raise ValueError(f"Unsupported file format: '{extension}'")

        # Sanitize against prompt injections and malicious system tags
        sanitized_text = SafetyGuardrails.sanitize_document_text(extracted_text)
        return sanitized_text

    @staticmethod
    def chunk_text(
        text: str,
        source_name: str,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> List[Dict[str, Any]]:
        """
        Splits raw text into overlapping semantic passages with source metadata.
        """
        if not text:
            return []

        chunks: List[Dict[str, Any]] = []
        start = 0
        text_length = len(text)
        chunk_idx = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_slice = text[start:end]

            # Try to break at a paragraph or sentence boundary if possible
            if end < text_length:
                last_newline = chunk_slice.rfind("\n")
                last_period = chunk_slice.rfind(". ")
                break_point = max(last_newline, last_period)

                if break_point > chunk_size // 2:
                    end = start + break_point + 1
                    chunk_slice = text[start:end]

            cleaned_chunk = chunk_slice.strip()
            if len(cleaned_chunk) > 20:  # Ignore trivial fragment chunks
                chunks.append(
                    {
                        "chunk_id": chunk_idx,
                        "source": source_name,
                        "text": cleaned_chunk,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                chunk_idx += 1

            start = end - chunk_overlap
            if start >= text_length or end >= text_length:
                break

        return chunks
